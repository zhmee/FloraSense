"""
Query-aware explanation generation for flower recommendations.

The recommender already retrieves and ranks with latent semantic vectors. This
module turns that retrieval evidence into short user-facing explanations. If an
API key is configured, the explanation text is rewritten by the class LLM
client; otherwise the deterministic fallback still uses the same evidence.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

MAX_FLOWERS_PER_LLM_CALL = 10
MAX_EVIDENCE_VALUES = 4
QUERY_STOPWORDS = {
    "about",
    "after",
    "best",
    "flower",
    "flowers",
    "for",
    "from",
    "give",
    "have",
    "that",
    "the",
    "this",
    "want",
    "with",
    "mean",
    "meaning",
    "meanings",
    "means",
    "represent",
    "represents",
    "symbol",
    "symbolize",
    "symbolizes",
}
MEANING_INTENT_TOKENS = {
    "courage",
    "friendship",
    "gratitude",
    "hope",
    "love",
    "meaning",
    "means",
    "remembrance",
    "represent",
    "represents",
    "strength",
    "symbol",
    "symbolize",
    "symbolizes",
}
OCCASION_WORDS = {
    "anniversary": "anniversary",
    "birthday": "birthday",
    "birthdays": "birthday",
    "celebration": "celebrations",
    "celebrations": "celebrations",
    "christmas": "Christmas",
    "easter": "Easter",
    "funeral": "funeral",
    "funerals": "funeral",
    "graduation": "graduations",
    "graduations": "graduations",
    "mother": "Mother's Day",
    "sympathy": "sympathy",
    "valentine": "Valentine's Day",
    "wedding": "wedding",
    "weddings": "wedding",
}
USER_FACING_BLOCKED_TERMS = (
    "axis",
    "database",
    "latent",
    "matched keyword",
    "retrieval",
    "retrieved",
    "score",
    "vector",
)
MISSING_VALUE_LABELS = {"", "n/a", "na", "none", "not listed", "null", "unknown"}
QUERY_TOKEN_ALIASES = {
    "maintanence": "maintenance",
    "maintenence": "maintenance",
    "maintainance": "maintenance",
    "miantenance": "maintenance",
}
NEGATED_HIGH_MAINTENANCE_TEXT = (
    r"\b(?:not|no|avoid|without)\s+(?:a\s+)?high\s+maintenance\b",
    r"\b(?:not|no|avoid|without)\s+(?:a\s+)?high\s+care\b",
    r"\b(?:not|no|avoid|without)\s+(?:a\s+)?high\s+upkeep\b",
    r"\bnot\s+hard\s+to\s+(?:maintain|care\s+for|grow)\b",
    r"\bnot\s+(?:a\s+)?lot\s+of\s+care\b",
    r"\b(?:don['’]?t|do\s+not|doesn['’]?t|does\s+not|shouldn['’]?t|should\s+not|can['’]?t|cannot|can\s+not)\s+(?:require|need|take|demand)\s+(?:too\s+much|that\s+much|much|a\s+lot\s+of|lots\s+of)\s+(?:care|maintenance|upkeep|attention|effort)\b",
    r"\b(?:not|without)\s+(?:too\s+much|that\s+much|much|a\s+lot\s+of|lots\s+of)\s+(?:care|maintenance|upkeep|attention|effort)\b",
)
NEGATED_LOW_MAINTENANCE_TEXT = (
    r"\b(?:not|no|avoid|without)\s+(?:a\s+)?low\s+maintenance\b",
    r"\b(?:not|no|avoid|without)\s+(?:an\s+)?easy\s+(?:care|maintenance|upkeep)\b",
    r"\bnot\s+easy\s+to\s+(?:maintain|care\s+for|grow)\b",
)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _canonicalize_query_terms(query: str) -> str:
    canonical_query = query
    for typo, replacement in QUERY_TOKEN_ALIASES.items():
        canonical_query = re.sub(rf"\b{re.escape(typo)}\b", replacement, canonical_query, flags=re.IGNORECASE)
    return canonical_query


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _is_missing_value(value: str) -> bool:
    return _clean_text(value).lower() in MISSING_VALUE_LABELS


def _sentence_case(value: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = _clean_text(value)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _compact_values(values: Any, limit: int = MAX_EVIDENCE_VALUES) -> list[str]:
    if not isinstance(values, list):
        return []
    return _dedupe([
        _clean_text(value)
        for value in values
        if not _is_missing_value(value)
    ])[:limit]


def _compact_evidence_labels(values: Any, limit: int = MAX_EVIDENCE_VALUES) -> list[str]:
    if not isinstance(values, list):
        return []

    labels = []
    for value in values:
        for part in re.split(r"[;\n]+", _clean_text(value)):
            cleaned = part.strip(" .,:")
            if not cleaned or _is_missing_value(cleaned):
                continue
            word_count = len(cleaned.split())
            has_sentence_punctuation = bool(re.search(r"[.!?]", cleaned))
            if word_count > 8 or len(cleaned) > 80 or has_sentence_punctuation:
                continue
            labels.append(cleaned)

    return _dedupe(labels)[:limit]


def _query_signal_tokens(query: str) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_key(query))
        if len(token) >= 4 and token not in QUERY_STOPWORDS
    ]
    return _dedupe(tokens)


def _query_categories(query_keywords: Any) -> set[str]:
    if not isinstance(query_keywords, list):
        return set()

    return {
        _clean_text(item.get("category"))
        for item in query_keywords
        if isinstance(item, dict) and _clean_text(item.get("category"))
    }


def _has_negated_high_maintenance(query: str) -> bool:
    canonical_query = _canonicalize_query_terms(query)
    return any(
        re.search(pattern, canonical_query, flags=re.IGNORECASE)
        for pattern in NEGATED_HIGH_MAINTENANCE_TEXT
    )


def _has_negated_low_maintenance(query: str) -> bool:
    canonical_query = _canonicalize_query_terms(query)
    return any(
        re.search(pattern, canonical_query, flags=re.IGNORECASE)
        for pattern in NEGATED_LOW_MAINTENANCE_TEXT
    )


def _query_breakdown_explanation(query: str, item: dict) -> str:
    keyword = _clean_text(item.get("keyword"))
    category = _clean_text(item.get("category"))
    normalized_keyword = _normalize_key(keyword)

    if category == "maintenance":
        if normalized_keyword == "low maintenance" and _has_negated_high_maintenance(query):
            return "Interprets your negative wording as a request for easier-care flowers."
        if normalized_keyword == "high maintenance" and _has_negated_low_maintenance(query):
            return "Interprets your negative wording as a request for flowers that need more care."
        if normalized_keyword == "high maintenance":
            return "Prioritizes flowers suited for more involved care."
        if normalized_keyword == "low maintenance":
            return "Prioritizes flowers with easier day-to-day care."
        if normalized_keyword:
            return f"Uses the requested {keyword} care profile."

    if category == "color":
        return f"Narrows results to flowers available in {keyword}."
    if category == "occasion":
        return f"Looks for flowers suited to {keyword}."
    if category == "meaning":
        return f"Favors flowers whose symbolism speaks to {keyword}."
    if category == "plant_type":
        return f"Uses the requested plant type: {keyword}."

    return "Query clue used for matching."


def _add_query_breakdown_explanations(payload: dict) -> list[dict]:
    query = _clean_text(payload.get("query"))
    explained_keywords = []
    for item in payload.get("keywords_used", []) or []:
        keyword_item = dict(item)
        keyword_item["explanation"] = _query_breakdown_explanation(query, keyword_item)
        keyword_item["explanation_source"] = "local"
        explained_keywords.append(keyword_item)

    return explained_keywords


def _query_terms_by_category(query_keywords: Any) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    if not isinstance(query_keywords, list):
        return grouped

    for item in query_keywords:
        if not isinstance(item, dict):
            continue
        category = _clean_text(item.get("category"))
        keyword = _clean_text(item.get("keyword"))
        if not category or not keyword:
            continue
        grouped.setdefault(category, []).append(keyword)

    return {
        category: _dedupe(values)
        for category, values in grouped.items()
    }


def _query_has_meaning_intent(query: str, categories: set[str]) -> bool:
    if "meaning" in categories:
        return True
    query_tokens = set(re.findall(r"[a-z0-9]+", _normalize_key(query)))
    return bool(query_tokens & MEANING_INTENT_TOKENS)


def _values_mentioned_by_query(query: str, values: Any, limit: int = MAX_EVIDENCE_VALUES) -> list[str]:
    if not isinstance(values, list):
        return []

    normalized_query = _normalize_key(query)
    if not normalized_query:
        return []

    matches = []
    for value in values:
        cleaned = _clean_text(value)
        normalized_value = _normalize_key(cleaned)
        if not cleaned or not normalized_value or _is_missing_value(cleaned):
            continue
        if re.search(rf"(?<!\w){re.escape(normalized_value)}(?!\w)", normalized_query):
            matches.append(cleaned)

    return _dedupe(matches)[:limit]


def _matching_query_terms(query_terms: list[str], suggestion_values: Any, limit: int = MAX_EVIDENCE_VALUES) -> list[str]:
    if not isinstance(suggestion_values, list):
        return []

    normalized_values = {
        _normalize_key(value)
        for value in suggestion_values
        if _normalize_key(value)
    }
    matches = []
    for term in query_terms:
        normalized_term = _normalize_key(term)
        if not normalized_term:
            continue
        if any(normalized_term == value or normalized_term in value or value in normalized_term for value in normalized_values):
            matches.append(term)

    return _dedupe(matches)[:limit]


def _extract_phrase_around_token(text: str, token: str, window: int = 3) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", text)
    if not words:
        return ""

    normalized_words = [_normalize_key(word) for word in words]
    for index, word in enumerate(normalized_words):
        if word != token:
            continue

        next_word = normalized_words[index + 1] if index + 1 < len(normalized_words) else ""
        second_next_word = normalized_words[index + 2] if index + 2 < len(normalized_words) else ""
        if next_word in {
            "arrangement",
            "arrangements",
            "bouquet",
            "bouquets",
            "flowers",
            "gift",
            "gifts",
            "anniversary",
        }:
            phrase_words = [words[index], words[index + 1]]
            if next_word == "floral" and second_next_word in {"arrangement", "arrangements"}:
                phrase_words.append(words[index + 2])
            return " ".join(phrase_words).strip(" .,:;-")

        if next_word == "floral" and second_next_word in {"arrangement", "arrangements"}:
            return " ".join(words[index : index + 3]).strip(" .,:;-")

        if index > 0 and normalized_words[index - 1] not in {"a", "an", "at", "for", "in", "of", "the"}:
            previous_word = words[index - 1].strip(" .,:;-")
            if previous_word and _normalize_key(previous_word) not in QUERY_STOPWORDS:
                return f"{previous_word} {words[index]}".strip(" .,:;-")

        return words[index].strip(" .,:;-")

    return ""


def _known_occasion_labels(values: Any, limit: int = MAX_EVIDENCE_VALUES) -> list[str]:
    if not isinstance(values, list):
        return []

    labels = []
    for value in values:
        normalized_value = _normalize_key(_clean_text(value))
        if not normalized_value or _is_missing_value(value):
            continue
        for token, label in OCCASION_WORDS.items():
            if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", normalized_value):
                labels.append(label)

        graduation_heading = re.search(
            r"\bgraduations?\s+and\s+celebrations?\s+of\s+achievement\b",
            normalized_value,
        )
        if graduation_heading:
            labels.append("graduations and achievements")

    return _dedupe(labels)[:limit]


def _occasion_evidence_labels(query: str, values: Any, limit: int = MAX_EVIDENCE_VALUES) -> list[str]:
    labels = _compact_evidence_labels(values, limit)
    if len(labels) >= limit:
        return labels[:limit]
    if not isinstance(values, list):
        return labels

    query_tokens = _query_signal_tokens(query)
    if not query_tokens:
        return _dedupe(labels + _known_occasion_labels(values, limit))[:limit]

    phrase_candidates = []
    for value in values:
        for chunk in re.split(r"(?<=[.!?])\s+|[;\n]+", _clean_text(value)):
            cleaned = chunk.strip(" .,:")
            normalized_chunk = _normalize_key(cleaned)
            if not cleaned or not normalized_chunk or _is_missing_value(cleaned):
                continue

            for token in query_tokens:
                if not re.search(rf"(?<!\w){re.escape(token)}(?!\w)", normalized_chunk):
                    continue

                prefix_match = re.match(r"^([A-Za-z0-9'’ -]{4,48}?)(?:\s+[-:–—]\s+|,)", cleaned)
                if prefix_match:
                    phrase_candidates.append(prefix_match.group(1).strip(" .,:;-"))

                phrase = _extract_phrase_around_token(cleaned, token)
                if phrase:
                    phrase_candidates.append(phrase)

    return _dedupe(labels + phrase_candidates + _known_occasion_labels(values, limit))[:limit]


def _join_human(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _shorten_words(text: str, max_words: int = 42) -> str:
    words = _clean_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(" ,;:.") + "."


def _is_user_facing_text(text: str) -> bool:
    normalized = _clean_text(text).lower()
    return bool(normalized) and not any(term in normalized for term in USER_FACING_BLOCKED_TERMS)


def _keyword_labels(suggestion: dict) -> list[str]:
    labels = []
    for match in suggestion.get("matched_keywords", []) or []:
        keyword = _clean_text(match.get("keyword"))
        category = _clean_text(match.get("category"))
        if not keyword:
            continue
        labels.append(f"{keyword} ({category})" if category else keyword)
    return _dedupe(labels)[:5]


def _keyword_terms(suggestion: dict, categories: set[str] | None = None) -> list[str]:
    terms = []
    for match in suggestion.get("matched_keywords", []) or []:
        keyword = _clean_text(match.get("keyword"))
        category = _clean_text(match.get("category"))
        if not keyword:
            continue
        if categories is not None and category not in categories:
            continue
        terms.append(keyword)
    return _dedupe(terms)[:5]


def _query_term_supported(term: str, values: Any) -> bool:
    normalized_term = _normalize_key(term)
    if not normalized_term or not isinstance(values, list):
        return False
    return any(
        normalized_term == _normalize_key(value)
        or normalized_term in _normalize_key(value)
        or _normalize_key(value) in normalized_term
        for value in values
        if _normalize_key(value)
    )


def _query_focus_terms(query: str) -> list[str]:
    return [
        token
        for token in _query_signal_tokens(query)
        if token not in {"care", "easy", "hard", "need", "needs", "want"}
    ][:4]


def _fallback_explanation(query: str, suggestion: dict, query_keywords: Any = None) -> str:
    name = _clean_text(suggestion.get("name")) or "This flower"
    categories = _query_categories(query_keywords)
    query_terms = _query_terms_by_category(query_keywords)
    meaning_intent = _query_has_meaning_intent(query, categories)
    meaning_terms = _matching_query_terms(
        query_terms.get("meaning", []),
        suggestion.get("meanings"),
        3,
    )
    occasion_terms = _matching_query_terms(
        query_terms.get("occasion", []),
        suggestion.get("occasions"),
        2,
    )
    meanings = _compact_evidence_labels(suggestion.get("meanings"), 3)
    maintenance = _compact_values(suggestion.get("maintenance"), 1)
    colors = _matching_query_terms(query_terms.get("color", []), suggestion.get("colors"), 2)
    if not colors:
        colors = _values_mentioned_by_query(query, suggestion.get("colors"), 2)
    maintenance_terms = _matching_query_terms(
        query_terms.get("maintenance", []),
        suggestion.get("maintenance"),
        1,
    )
    plant_types = _matching_query_terms(query_terms.get("plant_type", []), suggestion.get("plant_types"), 2)
    if not plant_types:
        plant_types = _values_mentioned_by_query(query, suggestion.get("plant_types"), 2)

    focus_terms = _query_focus_terms(query)
    supported_focus = [
        term
        for term in focus_terms
        if (
            _query_term_supported(term, suggestion.get("colors"))
            or _query_term_supported(term, suggestion.get("meanings"))
            or _query_term_supported(term, suggestion.get("occasions"))
            or _query_term_supported(term, suggestion.get("plant_types"))
            or _query_term_supported(term, suggestion.get("maintenance"))
        )
    ]
    unsupported_focus = [term for term in focus_terms if term not in supported_focus and term not in {"flower", "flowers"}]

    reasons = []
    if colors:
        reasons.append(f"matches the requested {_join_human(colors)} color")
    if maintenance_terms:
        reasons.append(f"fits the requested {_join_human(maintenance_terms)} care level")
    elif maintenance and ("maintenance" in categories or _values_mentioned_by_query(query, maintenance, 1)):
        reasons.append(f"has {maintenance[0]} care needs")
    if plant_types:
        reasons.append(f"is a {_join_human(plant_types)}")
    if meaning_terms:
        reasons.append(f"directly supports {_join_human(meaning_terms)}")
    if occasion_terms:
        reasons.append(f"suits {_join_human(occasion_terms[:2])}")

    query_clause = f' for "{_clean_text(query)}"' if _clean_text(query) else ""
    if not reasons:
        return _sentence_case(
            _shorten_words(f"{name} is a weaker fit{query_clause}; the visible flower details only partially support the request.")
        )

    if unsupported_focus and reasons:
        return _sentence_case(
            _shorten_words(
                f"{name} is a partial fit{query_clause}: it {_join_human(reasons)}, but the visible evidence does not clearly support {_join_human(unsupported_focus[:2])}.",
                44,
            )
        )

    return _sentence_case(_shorten_words(f"{name} works{query_clause} because it {_join_human(reasons)}.", 38))


def _fallback_occasion_summary(query: str, suggestion: dict) -> str:
    name = _clean_text(suggestion.get("name")) or "This flower"
    occasions = _occasion_evidence_labels(query, suggestion.get("occasions"), 3)
    keywords = _keyword_terms(suggestion, {"meaning", "occasion"})[:3]

    if not occasions:
        return ""

    occasion_clause = _join_human(occasions)
    query_tokens = set(_query_signal_tokens(query))
    has_occasion_intent = bool(query_tokens & set(OCCASION_WORDS)) or any(
        _normalize_key(label) in query_tokens
        for label in occasions
    )
    if not has_occasion_intent:
        return _sentence_case(_shorten_words(f"{occasion_clause}.", 12))

    query_clause = f' for "{_clean_text(query)}"' if _clean_text(query) else " for this request"
    if keywords:
        return _sentence_case(
            _shorten_words(
                f"{name} is a good occasion fit{query_clause} when the moment is {occasion_clause}; its symbolism supports {_join_human(keywords)}.",
                44,
            )
        )

    return _sentence_case(
        _shorten_words(
            f"{name} is a good occasion fit{query_clause} when the moment is {occasion_clause}.",
            44,
        )
    )


def _llm_explanations_enabled() -> bool:
    value = os.getenv("FLORASENSE_LLM_EXPLANATIONS", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def _llm_client():
    if not _llm_explanations_enabled():
        return None

    api_key = os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        return None

    try:
        from infosci_spark_client import LLMClient
    except Exception:
        logger.exception("Could not import infosci_spark_client for flower explanations.")
        return None

    return LLMClient(api_key=api_key)


def _build_llm_payload(payload: dict) -> dict:
    query = _clean_text(payload.get("query"))
    query_keywords = [
        {
            "keyword": _clean_text(item.get("keyword")),
            "category": _clean_text(item.get("category")),
        }
        for item in (payload.get("keywords_used", []) or [])[:8]
        if _clean_text(item.get("keyword"))
    ]

    flowers = []
    for suggestion in (payload.get("suggestions", []) or [])[:MAX_FLOWERS_PER_LLM_CALL]:
        flowers.append(
            {
                "name": _clean_text(suggestion.get("name")),
                "scientific_name": _clean_text(suggestion.get("scientific_name")),
                "score": suggestion.get("score"),
                "supporting_terms": [
                    {
                        "keyword": _clean_text(match.get("keyword")),
                        "category": _clean_text(match.get("category")),
                    }
                    for match in (suggestion.get("matched_keywords", []) or [])[:6]
                    if _clean_text(match.get("keyword"))
                ],
                "meaning_evidence": _compact_evidence_labels(suggestion.get("meanings"), 4),
                "occasion_evidence": _occasion_evidence_labels(query, suggestion.get("occasions"), 3),
                "colors": _compact_values(suggestion.get("colors"), 4),
                "maintenance": _compact_values(suggestion.get("maintenance"), 2),
                "plant_types": _compact_values(suggestion.get("plant_types"), 3),
            }
        )

    return {
        "query": query,
        "query_keywords": query_keywords,
        "flowers": flowers,
    }


def _extract_json(content: str) -> Any:
    content = _clean_text(content)
    if not content:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    array_start = content.find("[")
    array_end = content.rfind("]")
    if array_start != -1 and array_end > array_start:
        try:
            return json.loads(content[array_start : array_end + 1])
        except json.JSONDecodeError:
            return None

    object_start = content.find("{")
    object_end = content.rfind("}")
    if object_start != -1 and object_end > object_start:
        try:
            return json.loads(content[object_start : object_end + 1])
        except json.JSONDecodeError:
            return None

    return None


def _normalize_llm_items(parsed: Any) -> list[dict]:
    if isinstance(parsed, dict):
        if isinstance(parsed.get("explanations"), list):
            return parsed["explanations"]
        if isinstance(parsed.get("suggestions"), list):
            return parsed["suggestions"]
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def _generate_llm_explanations(payload: dict) -> dict[str, dict[str, str]]:
    client = _llm_client()
    if client is None:
        return {}

    llm_payload = _build_llm_payload(payload)
    if not llm_payload["flowers"]:
        return {}

    messages = [
        {
            "role": "system",
            "content": (
                "You write concise FloraSense recommendation explanations. "
                "Use only the provided meaning evidence, occasion evidence, supporting terms, "
                "colors, maintenance, and plant types. Do not invent flower symbolism. "
                "Prioritize the user's query keywords and matching structured attributes first; "
                "then use the flower's meaning evidence as support when the query is about symbolism. "
                "If the query is not about meaning or symbolism, do not mention symbolic meanings. "
                "Explain why each flower works for the user's query in one polished sentence "
                "of 22 to 45 words. Also write a separate occasion summary of 12 to 32 words "
                "using only provided occasion evidence; use an empty string if no occasion evidence is present. "
                "Write like a florist explaining the recommendation. Never mention retrieval, data, "
                "matched keywords, scores, latent space, vectors, axes, or the model. "
                "Return JSON only: "
                "[{\"name\":\"Flower name\",\"query_fit_explanation\":\"sentence\","
                "\"query_fit_occasion_summary\":\"sentence or empty string\"}]."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(llm_payload, ensure_ascii=True),
        },
    ]

    try:
        response = client.chat(messages)
    except Exception:
        logger.exception("LLM flower explanation generation failed.")
        return {}

    parsed = _extract_json((response or {}).get("content", ""))
    explanations = {}
    for item in _normalize_llm_items(parsed):
        if not isinstance(item, dict):
            continue
        name = _clean_text(item.get("name"))
        explanation = _shorten_words(_clean_text(item.get("query_fit_explanation")), 52)
        occasion_summary = _shorten_words(_clean_text(item.get("query_fit_occasion_summary")), 36)
        if explanation and not _is_user_facing_text(explanation):
            explanation = ""
        if occasion_summary and not _is_user_facing_text(occasion_summary):
            occasion_summary = ""
        if not name or (not explanation and not occasion_summary):
            continue
        explanations[_normalize_key(name)] = {
            "query_fit_explanation": explanation,
            "query_fit_occasion_summary": occasion_summary,
        }

    return explanations


def add_query_fit_explanations(payload: dict) -> dict:
    """Attach a query-aware explanation to each recommendation suggestion."""
    normalized = dict(payload or {})
    query = _clean_text(normalized.get("query"))
    normalized["keywords_used"] = _add_query_breakdown_explanations(normalized)
    llm_explanations = _generate_llm_explanations(normalized)

    suggestions = []
    for suggestion in normalized.get("suggestions", []) or []:
        item = dict(suggestion)
        name_key = _normalize_key(_clean_text(item.get("name")))
        fallback = _fallback_explanation(query, item, normalized.get("keywords_used", []))
        fallback_occasion = _fallback_occasion_summary(query, item)
        llm_item = llm_explanations.get(name_key, {})
        llm_text = llm_item.get("query_fit_explanation", "") if isinstance(llm_item, dict) else ""
        llm_occasion = llm_item.get("query_fit_occasion_summary", "") if isinstance(llm_item, dict) else ""
        item["query_fit_explanation"] = llm_text or fallback
        item["explanation_source"] = "llm" if llm_text else "local"
        item["query_fit_occasion_summary"] = llm_occasion or fallback_occasion
        item["occasion_summary_source"] = "llm" if llm_occasion else ("local" if fallback_occasion else "")
        suggestions.append(item)

    normalized["suggestions"] = suggestions
    return normalized
