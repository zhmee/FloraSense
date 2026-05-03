"""
Query-aware metadata helpers for flower recommendations.

Meaning and occasion card text is preprocessed in
`merged_preprocessed_meanings.csv`; this module no longer rewrites that copy or
fills missing meaning/occasion text.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

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
NEGATIVE_LOVE_MEANING_PATTERNS = (
    "decrease of love",
    "fading love",
    "lost love",
    "love denied",
    "rejected love",
    "unrequited love",
    "infidelity",
    "betrayal",
)
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
    "mothers": "Mother's Day",
    "mom": "Mother's Day",
    "moms": "Mother's Day",
    "mum": "Mother's Day",
    "mums": "Mother's Day",
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
    text = (
        str(value or "")
        .replace("�", "'")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("–", "-")
        .replace("—", "-")
    )
    return re.sub(r"\s+", " ", text).strip()


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


def _meaning_value_supports_query_term(term: str, value: str) -> bool:
    normalized_term = _normalize_key(term)
    normalized_value = _normalize_key(value)
    if not normalized_term or not normalized_value:
        return False
    if any(pattern in normalized_term for pattern in NEGATIVE_LOVE_MEANING_PATTERNS):
        return False
    if normalized_term == "love" and any(pattern in normalized_value for pattern in NEGATIVE_LOVE_MEANING_PATTERNS):
        return False
    return normalized_term == normalized_value or normalized_term in normalized_value or normalized_value in normalized_term


def _matching_meaning_query_terms(query_terms: list[str], suggestion_values: Any, limit: int = MAX_EVIDENCE_VALUES) -> list[str]:
    if not isinstance(suggestion_values, list):
        return []

    matches = []
    for term in query_terms:
        if any(_meaning_value_supports_query_term(term, value) for value in suggestion_values):
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


def _occasion_label_matches_query(label: str, query_tokens: set[str]) -> bool:
    normalized_label = _normalize_key(label)
    if not normalized_label or not query_tokens:
        return False
    label_tokens = set(re.findall(r"[a-z0-9]+", normalized_label))
    if label == "Mother's Day":
        return bool(query_tokens & {"mother", "mothers", "mom", "moms", "mum", "mums"})
    return bool(label_tokens & query_tokens)


def _is_usable_occasion_phrase(value: str) -> bool:
    normalized = _normalize_key(value)
    if not normalized:
        return False
    if normalized.startswith(("the flower", "its ", "this ", "that ", "their ")):
        return False
    return any(
        re.search(rf"(?<!\w){re.escape(token)}(?!\w)", normalized)
        for token in OCCASION_WORDS
    )


def _expanded_occasion_query_tokens(query_tokens: set[str]) -> set[str]:
    tokens = set(query_tokens)
    if tokens & {"mother", "mothers", "mom", "moms", "mum", "mums"}:
        tokens.update({"mother", "mothers", "mom", "moms", "mum", "mums"})
    return tokens


def _occasion_detail_clause(query: str, values: Any) -> str:
    if not isinstance(values, list):
        return ""

    query_tokens = _expanded_occasion_query_tokens(set(_query_signal_tokens(query)))
    if not query_tokens:
        return ""

    for value in values:
        for chunk in re.split(r"(?<=[.!?])\s+|[;\n]+", _clean_text(value)):
            cleaned = chunk.strip(" .,:")
            normalized_chunk = _normalize_key(cleaned)
            if not cleaned or not normalized_chunk or _is_missing_value(cleaned):
                continue
            if not any(
                re.search(rf"(?<!\w){re.escape(token)}(?!\w)", normalized_chunk)
                for token in query_tokens
            ):
                continue

            detail = cleaned
            if ":" in detail:
                heading, remainder = [part.strip(" .,:;-") for part in detail.split(":", 1)]
                if any(
                    re.search(rf"(?<!\w){re.escape(token)}(?!\w)", _normalize_key(heading))
                    for token in query_tokens
                ) and remainder:
                    detail = remainder

            detail = re.sub(r"^offer\s+", "", detail, flags=re.IGNORECASE).strip(" .,:;-")
            detail = re.sub(r"^as\s+", "", detail, flags=re.IGNORECASE).strip(" .,:;-")
            detail = re.sub(r"\bthe flower['’]s\b", "its", detail, flags=re.IGNORECASE)
            detail = re.sub(r"\bits symbolism\b", "its symbolism", detail, flags=re.IGNORECASE)
            before_dash, dash, after_dash = detail.partition(" - ")
            if dash and after_dash and any(
                re.search(rf"(?<!\w){re.escape(token)}(?!\w)", _normalize_key(before_dash))
                for token in query_tokens
            ):
                detail = after_dash.strip(" .,:;-")
            detail = re.sub(
                r",?\s+from\s+(mother's day)\s+to\s+[^.]+",
                r", including \1",
                detail,
                flags=re.IGNORECASE,
            )
            if re.search(r"\bmother['’]?s day surprise sorted\??$", detail, flags=re.IGNORECASE):
                detail = "it is presented as a Mother's Day surprise choice"
            for separator in (" while ", " however, ", " needless to say, "):
                before, found, _after = detail.partition(separator)
                if found and any(
                    re.search(rf"(?<!\w){re.escape(token)}(?!\w)", _normalize_key(before))
                    for token in query_tokens
                ):
                    detail = before.strip(" .,:;-")
                    break
            if not detail:
                continue
            return _shorten_words(detail, 28).rstrip(".")

    return ""


def _occasion_evidence_labels(query: str, values: Any, limit: int = MAX_EVIDENCE_VALUES) -> list[str]:
    labels = _compact_evidence_labels(values, limit)
    if len(labels) >= limit:
        return labels[:limit]
    if not isinstance(values, list):
        return labels

    query_tokens = _query_signal_tokens(query)
    known_labels = _known_occasion_labels(values, limit=12)
    if not query_tokens:
        return _dedupe(labels + known_labels)[:limit]

    query_token_set = _expanded_occasion_query_tokens(set(query_tokens))
    query_matched_known_labels = [
        label
        for label in known_labels
        if _occasion_label_matches_query(label, query_token_set)
    ]
    if query_matched_known_labels:
        return _dedupe(query_matched_known_labels + labels)[:limit]

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
                if prefix_match and _is_usable_occasion_phrase(prefix_match.group(1)):
                    phrase_candidates.append(prefix_match.group(1).strip(" .,:;-"))

                phrase = _extract_phrase_around_token(cleaned, token)
                if phrase and _is_usable_occasion_phrase(phrase):
                    phrase_candidates.append(phrase)

    return _dedupe(labels + phrase_candidates + known_labels)[:limit]


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


def _trim_dangling_tail(text: str) -> str:
    return re.sub(
        r"\s+(?:a|an|and|at|for|from|in|of|on|the|to|with)(?:\s+(?:a|an|the))?$",
        "",
        _clean_text(text),
        flags=re.IGNORECASE,
    ).strip(" ,;:.")


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


def add_query_fit_explanations(payload: dict, use_llm: bool = False) -> dict:
    """Attach query keyword explanations without rewriting card copy."""
    normalized = dict(payload or {})
    normalized["keywords_used"] = _add_query_breakdown_explanations(normalized)

    suggestions = []
    for suggestion in normalized.get("suggestions", []) or []:
        item = dict(suggestion)
        item["query_fit_explanation"] = item.get("query_fit_explanation", "")
        item["explanation_source"] = item.get("explanation_source", "")
        item["query_fit_occasion_summary"] = item.get("query_fit_occasion_summary", "")
        item["occasion_summary_source"] = item.get("occasion_summary_source", "")
        suggestions.append(item)

    normalized["suggestions"] = suggestions
    return normalized
