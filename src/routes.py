"""
Routes: React app serving and episode search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
#OLD VERSION
import json
import logging
import os
import re
import copy
import threading
import time
from functools import lru_cache
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from flask import send_from_directory, request, jsonify
from utils import FLOWER_IMAGE_DIR
from models import db, Episode, Review

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = True
# USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

VISUALIZATION_DIR = Path(__file__).resolve().parent / "3d_visualization"
try:
    LLM_MAX_CONCURRENCY = max(1, int(os.getenv("FLORASENSE_MAX_LLM_CONCURRENCY", "2")))
except ValueError:
    LLM_MAX_CONCURRENCY = 2
_LLM_SEMAPHORE = threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)


def json_search(query):
    if not query or not query.strip():
        query = "Kardashian"
    results = db.session.query(Episode, Review).join(
        Review, Episode.id == Review.id
    ).filter(
        Episode.title.ilike(f'%{query}%')
    ).all()
    matches = []
    for episode, review in results:
        matches.append({
            'title': episode.title,
            'descr': episode.descr,
            'imdb_rating': review.imdb_rating
        })
    return matches


def _load_python_source(module_name, source_path):
    loader = SourceFileLoader(module_name, str(source_path))
    spec = spec_from_loader(module_name, loader)
    if spec is None:
        raise ImportError(f"Could not load module spec for {source_path}")
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_search_modules():
    from flower_autocomplete import autocomplete_queries
    from flower_recommender_prototype4 import recommend_flowers, visualizer_flowers
    from flower_recommender_v3 import recommend_flowers_tfidf

    return {
        "recommend_flowers": recommend_flowers,
        "recommend_flowers_tfidf": recommend_flowers_tfidf,
        "visualizer_flowers": visualizer_flowers,
        "autocomplete_queries": autocomplete_queries,
    }


def _normalize_recommendation_payload(payload, query: str) -> dict:
    normalized = dict(payload or {})
    normalized["query"] = normalized.get("query", query)
    normalized["keywords_used"] = normalized.get("keywords_used", [])
    normalized["query_latent_radar_chart"] = normalized.get("query_latent_radar_chart")
    normalized["query_latent_radar_axes"] = normalized.get("query_latent_radar_axes", [])
    normalized["keywords_used"] = [
        {
            **dict(item),
            "explanation": dict(item).get("explanation", ""),
            "explanation_source": dict(item).get("explanation_source", ""),
        }
        for item in normalized["keywords_used"]
        if isinstance(item, dict)
    ]

    suggestions = []
    for suggestion in normalized.get("suggestions", []) or []:
        item = dict(suggestion)
        item["matched_keywords"] = item.get("matched_keywords", [])
        item["latent_radar_chart"] = item.get("latent_radar_chart")
        item["latent_radar_axes"] = item.get("latent_radar_axes", [])
        item["query_fit_explanation"] = item.get("query_fit_explanation", "")
        item["query_fit_occasion_summary"] = item.get("query_fit_occasion_summary", "")
        item["occasion_summary_source"] = item.get("occasion_summary_source", "")
        suggestions.append(item)
    normalized["suggestions"] = suggestions

    if "score_scale" not in normalized:
        scores = [
            float(suggestion.get("score", 0) or 0)
            for suggestion in suggestions
        ]
        normalized["score_scale"] = "unit" if scores and max(scores) <= 1.0 else "percent"

    return normalized


def _add_recommendation_explanations(payload: dict, use_llm: bool = False) -> dict:
    try:
        from flower_explanations import add_query_fit_explanations

        return add_query_fit_explanations(payload, use_llm=use_llm)
    except Exception:
        logger.exception("Could not build query-aware recommendation explanations.")
        return payload


def _recommend_with_fallback(query: str, limit: int, method: str, use_llm_explanations: bool = False) -> dict:
    modules = _load_search_modules()

    if method == "tfidf":
        payload = modules["recommend_flowers_tfidf"](query, limit=limit)
        return _add_recommendation_explanations(
            _normalize_recommendation_payload(payload, query),
            use_llm=use_llm_explanations,
        )

    try:
        payload = modules["recommend_flowers"](query, limit=limit)
        return _add_recommendation_explanations(
            _normalize_recommendation_payload(payload, query),
            use_llm=use_llm_explanations,
        )
    except Exception:
        logger.exception("SVD recommendations failed for query %r. Falling back to TF-IDF.", query)
        payload = modules["recommend_flowers_tfidf"](query, limit=limit)
        return _add_recommendation_explanations(
            _normalize_recommendation_payload(payload, query),
            use_llm=use_llm_explanations,
        )


@lru_cache(maxsize=256)
def _cached_recommend_with_fallback(query: str, limit: int, method: str, use_llm_explanations: bool = False) -> dict:
    return _recommend_with_fallback(query, limit, method, use_llm_explanations)


def _recommend_with_fallback_copy(query: str, limit: int, method: str, use_llm_explanations: bool = False) -> dict:
    return copy.deepcopy(
        _cached_recommend_with_fallback(query, limit, method, use_llm_explanations)
    )


def _llm_client():
    api_key = os.getenv("SPARK_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            logger.debug("python-dotenv is unavailable while loading Spark API key.", exc_info=True)
        api_key = os.getenv("SPARK_API_KEY")

    if not api_key:
        return None, "SPARK_API_KEY is not set."

    try:
        from infosci_spark_client import LLMClient
    except Exception:
        logger.exception("Could not import infosci_spark_client for RAG.")
        return None, "The LLM client could not be loaded."

    return LLMClient(api_key=api_key), ""


def _extract_json_object(content: str) -> dict:
    content = (content or "").strip()
    if not content:
        return {}

    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _llm_text_response(client, messages: list[dict]) -> str:
    try:
        response = client.chat(messages, stream=False, show_thinking=False)
    except TypeError:
        response = client.chat(messages)
    return (response or {}).get("content", "").strip()


QUERY_STOPWORDS = {
    "a",
    "about",
    "and",
    "any",
    "are",
    "actually",
    "because",
    "been",
    "basically",
    "but",
    "can",
    "could",
    "couldn",
    "couldnt",
    "do",
    "does",
    "doesn",
    "doesnt",
    "don",
    "doni",
    "dont",
    "for",
    "give",
    "have",
    "help",
    "i",
    "im",
    "in",
    "is",
    "isn",
    "isnt",
    "it",
    "just",
    "kinda",
    "kind",
    "like",
    "looking",
    "me",
    "maybe",
    "need",
    "needs",
    "not",
    "of",
    "or",
    "please",
    "quite",
    "really",
    "say",
    "show",
    "sort",
    "sorta",
    "someone",
    "something",
    "that",
    "the",
    "them",
    "they",
    "this",
    "to",
    "too",
    "very",
    "want",
    "wasn",
    "wasnt",
    "won",
    "wont",
    "with",
    "wouldn",
    "wouldnt",
    "who",
}
def _llm_retrieval_query(client, user_query: str) -> tuple[str, list[str], str, str]:
    messages = [
        {
            "role": "system",
            "content": (
                "Turn the user's request into a short flower search query for an IR system. "
                "Keep only searchable words: color, occasion, relationship, flower meaning, "
                "maintenance level, and flower names. Convert vague intent into simple flower "
                "meaning words such as gratitude, love, sympathy, friendship, remembrance, "
                "rebirth, strength, courage, low maintenance, or high maintenance. "
                "Remove filler words such as really, very, actually, basically, just, maybe, "
                "kind of, sort of, I, me, want, need, something, and please. "

                "NEGATION RULE: "
                "If the user negates any attribute (color, flower name, occasion, trait): "
                "  - OMIT that attribute from retrieval_query entirely. "
                "  - Add it to the exclude_terms list as a lowercase string. "
                "  - Only include attributes the user explicitly wants. "
                "If the user negates a trait with a clear opposite (e.g. 'not high maintenance'): "
                "  - Replace it with its opposite in retrieval_query (e.g. 'low maintenance'). "
                "  - Do NOT add it to exclude_terms in that case. "
                "Never include negation words (not, no, don't, without) in retrieval_query. "

                "Do not infer symbolic meaning from color constraints alone. "
                "Do not answer the user. Return JSON only: "
                "{\"retrieval_query\":\"short search phrase\","
                "\"exclude_terms\":[\"negated attributes as lowercase strings, or empty array\"],"
                "\"rationale\":\"brief reason\"}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Examples:\n"
                "User: I need flowers to say thanks to my teacher\n"
                "JSON: {\"retrieval_query\":\"gratitude teacher flowers\",\"exclude_terms\":[],\"rationale\":\"thanks maps to gratitude\"}\n"
                "User: something romantic but not too hard to take care of\n"
                "JSON: {\"retrieval_query\":\"love low maintenance flowers\",\"exclude_terms\":[],\"rationale\":\"not high maintenance flipped to low maintenance\"}\n"
                "User: I want flowers but not yellow and not roses\n"
                "JSON: {\"retrieval_query\":\"flowers\",\"exclude_terms\":[\"yellow\",\"rose\"],\"rationale\":\"yellow and rose excluded per user request\"}\n\n"
                f"User: {user_query}"
            ),
        },
    ]

    try:
        content = _llm_text_response(client, messages)
    except Exception:
        logger.exception("LLM query transformation failed.")
        return user_query, [], "LLM query transformation failed; using the original query.", "local"

    parsed = _extract_json_object(content)
    retrieval_query = str(parsed.get("retrieval_query") or "").strip()
    rationale = str(parsed.get("rationale") or "").strip()
    exclude_terms = [str(t).lower().strip() for t in parsed.get("exclude_terms") or [] if t]

    if not retrieval_query:
        return user_query, [], "The LLM did not return a valid retrieval query, so the original query was used.", "local"

    return retrieval_query, exclude_terms, rationale, "llm"

import re
def _apply_hard_filters(payload: dict, exclude_terms: list[str]) -> dict:
    if not exclude_terms:
        return payload

    filtered = []
    for suggestion in payload.get("suggestions", []):
        colors = [c.lower() for c in suggestion.get("colors", [])]
        name = suggestion.get("name", "").lower()

        # only exclude if the color field itself matches, not meanings/occasions
        color_excluded = colors and all(
            any(re.search(rf"\b{re.escape(term)}\b", c) for term in exclude_terms)
            for c in colors
        )
        name_excluded = any(
            re.search(rf"\b{re.escape(term)}\b", name)
            for term in exclude_terms
            if term not in {"white", "red", "pink", "yellow", "purple", "orange", "blue"}
        )

        if color_excluded or name_excluded:
            continue
        filtered.append(suggestion)

    if not filtered:
        return payload

    payload["suggestions"] = filtered
    return payload

def _compact_context_values(values, limit: int = 3) -> list[str]:
    if not isinstance(values, list):
        return []

    compacted = []
    seen = set()
    for value in values:
        text = " ".join(str(value or "").split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        compacted.append(text)
        if len(compacted) >= limit:
            break
    return compacted


def _rag_name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _build_rag_context_documents(payload: dict, limit: int = 5) -> list[dict]:
    documents = []
    for rank, suggestion in enumerate((payload.get("suggestions") or [])[:limit], start=1):
        matched_keywords = [
            {
                "keyword": match.get("keyword", ""),
                "category": match.get("category", ""),
            }
            for match in (suggestion.get("matched_keywords") or [])[:8]
            if isinstance(match, dict)
        ]
        documents.append(
            {
                "rank": rank,
                "name": suggestion.get("name", ""),
                "scientific_name": suggestion.get("scientific_name", ""),
                "score": suggestion.get("score"),
                "non_rag_explanation": suggestion.get("query_fit_explanation", ""),
                "non_rag_occasion_summary": suggestion.get("query_fit_occasion_summary", ""),
                "has_occasion_evidence": bool(
                    suggestion.get("query_fit_occasion_summary")
                    or suggestion.get("occasions")
                ),
                "colors": _compact_context_values(suggestion.get("colors"), 5),
                "maintenance": _compact_context_values(suggestion.get("maintenance"), 2),
                "plant_types": _compact_context_values(suggestion.get("plant_types"), 3),
                "meanings": _compact_context_values(suggestion.get("meanings"), 5),
                "occasions": _compact_context_values(suggestion.get("occasions"), 5),
                "matched_keywords": matched_keywords,
            }
        )
    return documents


def _format_context_list(label: str, values: list[str]) -> str:
    if not values:
        return f"- **{label}:** Not listed"
    return f"- **{label}:** {'; '.join(values)}"


def _format_flower_rag_context(context_documents: list[dict]) -> str:
    """Render retrieved flower records as Markdown context for the LLM.

    This mirrors the in-class RAG demo shape: retrieval happens first, then the
    full retrieved records are pasted into the final prompt as readable context.
    """
    parts = []
    for document in context_documents:
        matches = []
        for match in document.get("matched_keywords", []) or []:
            if not isinstance(match, dict):
                continue
            keyword = " ".join(str(match.get("keyword") or "").split())
            category = " ".join(str(match.get("category") or "").split())
            if keyword and category:
                matches.append(f"{keyword} ({category})")
            elif keyword:
                matches.append(keyword)

        lines = [
            f"### [{document.get('rank')}] {document.get('name') or 'Unnamed flower'}",
            f"- **Score:** {document.get('score')}",
            f"- **Scientific name:** {document.get('scientific_name') or 'Not listed'}",
            f"- **Has occasion evidence:** {'yes' if document.get('has_occasion_evidence') else 'no'}",
            _format_context_list("Colors", document.get("colors") or []),
            _format_context_list("Plant types", document.get("plant_types") or []),
            _format_context_list("Maintenance", document.get("maintenance") or []),
            _format_context_list("Matched query evidence", matches),
        ]

        non_rag_explanation = " ".join(
            str(document.get("non_rag_explanation") or "").split()
        )
        if non_rag_explanation:
            lines.extend(
                [
                    "",
                    '**Existing "Why this matches" card text to improve:**',
                    non_rag_explanation,
                ]
            )

        non_rag_occasion = " ".join(
            str(document.get("non_rag_occasion_summary") or "").split()
        )
        if non_rag_occasion:
            lines.extend(
                [
                    "",
                    "**Existing occasion fit text to improve:**",
                    non_rag_occasion,
                ]
            )

        meanings = document.get("meanings") or []
        if meanings:
            lines.extend(["", "**Meanings:**"])
            lines.extend(f"- {meaning}" for meaning in meanings)

        occasions = document.get("occasions") or []
        if occasions:
            lines.extend(["", "**Occasions:**"])
            lines.extend(f"- {occasion}" for occasion in occasions)

        parts.append("\n".join(lines))

    return "\n\n---\n\n".join(parts)


def _generate_rag_response(
    client,
    user_query: str,
    retrieval_query: str,
    context_documents: list[dict],
) -> tuple[str, dict[str, dict[str, str]]]:
    if not context_documents:
        return "I could not find matching flower records to ground an answer.", {}

    context_markdown = _format_flower_rag_context(context_documents)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the final reader in a retrieve-then-read flower recommendation system. "
                "Answer only from the retrieved flower records below. If a retrieved record only "
                "partly supports the user's request, say it is a partial fit instead of forcing a "
                "match. Do not invent meanings, colors, occasions, or care details. "
                "Write warm, polished florist-style prose, not a comma-separated inventory. "
                "The answer should compare the strongest flowers: explain what each is especially "
                "good for, and why someone might choose one over another. "
                "For every card summary, use the existing 'Why this matches' card text and "
                "occasion fit text as starting evidence, then rewrite both into one deeper "
                "user-facing explanation grounded in the retrieved record. Include why the flower "
                "matches the query, what evidence supports that match, and how its occasion fit "
                "matters when occasion evidence is present. If the existing text is awkward or "
                "thin, improve it with the listed colors, plant type, maintenance, meanings, "
                "occasions, and matched evidence. Do not copy weak phrases like "
                "'fits mother day', 'visible evidence', or 'symbolism supports' verbatim. "
                "For every card occasion summary, rewrite the existing occasion fit text into "
                "a graceful sentence about when the flower is suitable. If a record says it has "
                "occasion evidence, rag_occasion_summary must not be empty; return an empty "
                "string only when the record has no occasion evidence. "
                "Never mention RAG, IR,"
                "vectors, retrieval, database, matched keywords, score, or context. "
                "Return JSON only with this exact shape: "
                "{\"answer\":\"2 to 3 sentence overall answer\", "
                "\"cards\":[{\"rank\":1,\"name\":\"exact Flower name\","
                "\"scientific_name\":\"exact scientific name\","
                "\"rag_summary\":\"2 to 3 sentences grounded in the retrieved record\","
                "\"rag_occasion_summary\":\"one sentence about occasion fit or empty string\"}]}. "
                "The answer should be 2 to 3 graceful sentences, around 55 to 70 words total. "
                "The cards array must contain exactly one entry for every retrieved flower record, "
                "in the same order, using the exact rank, name, and scientific name shown. "
                "Do not skip duplicated or similar-looking records; each retrieved record needs "
                "its own rag_summary and rag_occasion_summary. "
                "Each rag_summary must be 2 to 3 useful sentences, 45 to 60 words total, and "
                "should explain the fit using evidence that matters for the original user query."
                "Include WHY it was chosen. Each "
                "rag_occasion_summary must be one useful sentence, 14 to 20 words, focused ONLY "
                "on occasion evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original user query:\n{user_query}\n\n"
                f"LLM-transformed query sent to the flower IR system:\n{retrieval_query}\n\n"
                f"Retrieved flower records:\n\n{context_markdown}"
            ),
        },
    ]

    try:
        content = _llm_text_response(client, messages)
    except Exception:
        logger.exception("LLM RAG answer generation failed.")
        return "", {}

    parsed = _extract_json_object(content)
    answer = str(parsed.get("answer") or "").strip()
    card_summaries: dict[str, dict[str, str]] = {}
    cards = parsed.get("cards")
    if isinstance(cards, list):
        for card in cards:
            if not isinstance(card, dict):
                continue
            rank = card.get("rank")
            try:
                rank_number = int(rank)
            except (TypeError, ValueError):
                rank_number = None
            name = str(card.get("name") or "").strip()
            scientific_name = str(card.get("scientific_name") or "").strip()
            summary = str(card.get("rag_summary") or "").strip()
            occasion_summary = str(card.get("rag_occasion_summary") or "").strip()
            card_text = {
                "rag_summary": summary,
                "rag_occasion_summary": occasion_summary,
            }
            if (summary or occasion_summary) and rank_number is not None:
                card_summaries[f"rank:{rank_number}"] = card_text
            if name and (summary or occasion_summary):
                card_summaries[_rag_name_key(name)] = card_text
            if scientific_name and (summary or occasion_summary):
                card_summaries[f"scientific:{_rag_name_key(scientific_name)}"] = card_text

    if not answer:
        answer = content

    return answer, card_summaries


def _generate_rag_overview_response(
    client,
    user_query: str,
    retrieval_query: str,
    context_documents: list[dict],
) -> str:
    if not context_documents:
        return "I could not find matching flower records to ground an answer."

    context_markdown = _format_flower_rag_context(context_documents[:3])
    messages = [
        {
            "role": "system",
            "content": (
                "You write only the short overview for a flower recommendation result. "
                "Use only the retrieved flower records. Do not invent meanings, colors, "
                "occasions, or care details. Compare the strongest one or two flowers and "
                "explain why they fit the user's request. Return plain text only, 2 sentences, "
                "about 45 to 60 words. Do not mention RAG, IR, vectors, retrieval, database, "
                "matched keywords, score, or context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original user query:\n{user_query}\n\n"
                f"Search query sent to the flower system:\n{retrieval_query}\n\n"
                f"Retrieved flower records:\n\n{context_markdown}"
            ),
        },
    ]

    try:
        return _llm_text_response(client, messages)
    except Exception:
        logger.exception("LLM RAG overview generation failed.")
        return ""


def _context_document_from_suggestion(suggestion: dict, rank: int = 1) -> dict:
    return {
        "rank": rank,
        "name": suggestion.get("name", ""),
        "scientific_name": suggestion.get("scientific_name", ""),
        "score": suggestion.get("score"),
        "non_rag_explanation": suggestion.get("query_fit_explanation", ""),
        "non_rag_occasion_summary": suggestion.get("query_fit_occasion_summary", ""),
        "has_occasion_evidence": bool(
            suggestion.get("query_fit_occasion_summary")
            or suggestion.get("occasions")
        ),
        "colors": _compact_context_values(suggestion.get("colors"), 5),
        "maintenance": _compact_context_values(suggestion.get("maintenance"), 2),
        "plant_types": _compact_context_values(suggestion.get("plant_types"), 3),
        "meanings": _compact_context_values(suggestion.get("meanings"), 5),
        "occasions": _compact_context_values(suggestion.get("occasions"), 5),
        "matched_keywords": [
            {
                "keyword": match.get("keyword", ""),
                "category": match.get("category", ""),
            }
            for match in (suggestion.get("matched_keywords") or [])[:8]
            if isinstance(match, dict)
        ],
    }


def _generate_rag_card_response(client, user_query: str, document: dict) -> dict[str, str]:
    context_markdown = _format_flower_rag_context([document])
    messages = [
        {
            "role": "system",
            "content": (
                "You write one refined flower recommendation card. Use only the retrieved "
                "flower record below. Do not invent meanings, colors, occasions, or care "
                "details. Return JSON only with this exact shape: "
                "{\"rag_summary\":\"2 concise sentences explaining why this flower fits\","
                "\"rag_occasion_summary\":\"one concise occasion sentence or empty string\"}. "
                "Do not mention RAG, IR, vectors, retrieval, database, matched keywords, "
                "score, or context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original user query:\n{user_query}\n\n"
                f"Retrieved flower record:\n\n{context_markdown}"
            ),
        },
    ]

    try:
        content = _llm_text_response(client, messages)
    except Exception:
        logger.exception("LLM RAG card generation failed.")
        return {}

    parsed = _extract_json_object(content)
    return {
        "rag_summary": str(parsed.get("rag_summary") or "").strip(),
        "rag_occasion_summary": str(parsed.get("rag_occasion_summary") or "").strip(),
    }


_RETRIEVAL_QUERY_CACHE: dict[str, tuple[str, list[str], str, str]] = {}
_RETRIEVAL_QUERY_CACHE_LOCK = threading.Lock()


def _cached_retrieval_query(query: str):
    with _RETRIEVAL_QUERY_CACHE_LOCK:
        cached = _RETRIEVAL_QUERY_CACHE.get(query)
    if cached is not None:
        return cached

    client, reason = _llm_client()
    if client is None:
        return query, [], reason, "local"
    if not _LLM_SEMAPHORE.acquire(blocking=False):
        return query, [], "LLM workers are busy; using the original query.", "local"
    try:
        result = _llm_retrieval_query(client, query)
        if result[3] == "llm":
            with _RETRIEVAL_QUERY_CACHE_LOCK:
                if len(_RETRIEVAL_QUERY_CACHE) >= 256:
                    _RETRIEVAL_QUERY_CACHE.pop(next(iter(_RETRIEVAL_QUERY_CACHE)))
                _RETRIEVAL_QUERY_CACHE[query] = result
        return result
    finally:
        _LLM_SEMAPHORE.release()

def _rag_recommendations(query: str, limit: int, method: str) -> dict:
    client, unavailable_reason = _llm_client()
    retrieval_query, exclude_terms, transform_rationale, transform_source = _cached_retrieval_query(query)

    payload = _recommend_with_fallback_copy(
        retrieval_query,
        limit,
        method,
        use_llm_explanations=False,
    )
   

    payload = _apply_hard_filters(payload, exclude_terms)

    if len(payload.get("suggestions", [])) < limit:
        logger.warning("Too few results after filtering, refilling...")
        modules = _load_search_modules()
        fallback = modules["recommend_flowers_tfidf"](retrieval_query, limit=limit * 2)
        fallback_payload = {"suggestions": fallback.get("suggestions", [])}
        fallback_payload = _apply_hard_filters(fallback_payload, exclude_terms)  # ← filter again
        payload["suggestions"] = fallback_payload.get("suggestions", [])[:limit]

    payload["query"] = query
    payload["query"] = query
    context_documents = _build_rag_context_documents(payload, limit=limit)

    answer = ""
    card_summaries = {}
    answer_source = "llm"
    card_summary_source = "llm"
    if client is not None:
        if _LLM_SEMAPHORE.acquire(blocking=False):
            try:
                answer, card_summaries = _generate_rag_response(
                    client,
                    query,
                    retrieval_query,
                    context_documents,
                )
            finally:
                _LLM_SEMAPHORE.release()
        else:
            logger.info("Skipping LLM answer generation because all LLM workers are busy.")

    if not answer:
        answer_source = ""
    if not card_summaries:
        card_summary_source = ""

    for suggestion_index, suggestion in enumerate(payload.get("suggestions", []) or [], start=1):
        name = suggestion.get("name", "")
        scientific_name = suggestion.get("scientific_name", "")
        name_key = _rag_name_key(name)
        scientific_key = f"scientific:{_rag_name_key(scientific_name)}"
        card_text = (
            card_summaries.get(f"rank:{suggestion_index}")
            or card_summaries.get(name_key)
            or card_summaries.get(scientific_key)
            or {}
        )
        summary = card_text.get("rag_summary", "")
        occasion_summary = card_text.get("rag_occasion_summary", "")

        suggestion["ir_summary"] = suggestion.get("ir_summary", "")
        suggestion["ir_summary_source"] = suggestion.get("ir_summary_source", "")
        suggestion["rag_summary"] = summary
        suggestion["rag_occasion_summary"] = occasion_summary
        suggestion["rag_source"] = card_summary_source if summary else ""
        suggestion["rag_occasion_source"] = card_summary_source if occasion_summary else ""
        suggestion["ir_query_fit_explanation"] = suggestion.get("query_fit_explanation", "")
        suggestion["ir_query_fit_occasion_summary"] = suggestion.get("query_fit_occasion_summary", "")

    payload["rag"] = {
        "user_query": query,
        "retrieval_query": retrieval_query,
        "query_transform_source": transform_source,
        "query_transform_rationale": transform_rationale,
        "answer": answer,
        "answer_source": answer_source,
        "context_documents": context_documents,
    }
    return payload


def _rag_overview(query: str, limit: int, method: str) -> dict:
    client, unavailable_reason = _llm_client()
    retrieval_query, exclude_terms, transform_rationale, transform_source = _cached_retrieval_query(query)
    payload = _recommend_with_fallback_copy(
        retrieval_query,
        limit,
        method,
        use_llm_explanations=False,
    )
    payload = _apply_hard_filters(payload, exclude_terms)
    payload["query"] = query
    context_documents = _build_rag_context_documents(payload, limit=min(limit, 3))

    answer = ""
    answer_source = "llm"
    if client is not None:
        if _LLM_SEMAPHORE.acquire(blocking=False):
            try:
                answer = _generate_rag_overview_response(
                    client,
                    query,
                    retrieval_query,
                    context_documents,
                )
            finally:
                _LLM_SEMAPHORE.release()
        else:
            logger.info("Skipping LLM overview because all LLM workers are busy.")
    else:
        logger.info("Skipping LLM overview: %s", unavailable_reason)

    if not answer:
        answer_source = ""

    return {
        "query": query,
        "rag": {
            "user_query": query,
            "retrieval_query": retrieval_query,
            "query_transform_source": transform_source,
            "query_transform_rationale": transform_rationale,
            "answer": answer,
            "answer_source": answer_source,
            "context_documents": context_documents,
        },
    }


def _rag_card_summary(query: str, suggestion: dict) -> dict:
    client, unavailable_reason = _llm_client()
    document = _context_document_from_suggestion(suggestion)

    card_text = {}
    card_summary_source = "llm"
    if client is not None:
        if _LLM_SEMAPHORE.acquire(blocking=False):
            try:
                card_text = _generate_rag_card_response(client, query, document)
            finally:
                _LLM_SEMAPHORE.release()
        else:
            logger.info("Skipping LLM card refinement because all LLM workers are busy.")
    else:
        logger.info("Skipping LLM card refinement: %s", unavailable_reason)

    if not any(card_text.values()):
        card_summary_source = ""

    return {
        "name": suggestion.get("name", ""),
        "scientific_name": suggestion.get("scientific_name", ""),
        "ir_summary": suggestion.get("ir_summary", ""),
        "ir_summary_source": suggestion.get("ir_summary_source", ""),
        "rag_summary": card_text.get("rag_summary", ""),
        "rag_occasion_summary": card_text.get("rag_occasion_summary", ""),
        "rag_source": card_summary_source if card_text.get("rag_summary") else "",
        "rag_occasion_source": card_summary_source if card_text.get("rag_occasion_summary") else "",
    }


@lru_cache(maxsize=1)
def _load_visualizer_insight_modules():
    return {
        "health": _load_python_source(
            "visualizer_health_bar_calculation",
            VISUALIZATION_DIR / "health_bar_calculation",
        ),
        "recommendations": _load_python_source(
            "visualizer_recommendation_calculation",
            VISUALIZATION_DIR / "recommendation_calculation",
        ),
    }


@lru_cache(maxsize=16)
def _cached_visualizer_flowers(limit: int) -> dict:
    modules = _load_search_modules()
    return modules["visualizer_flowers"](limit=limit)


@lru_cache(maxsize=256)
def _cached_visualizer_bouquet_insights(scientific_names: tuple[str, ...]) -> dict:
    modules = _load_visualizer_insight_modules()
    names = list(scientific_names)
    meanings_payload = modules["health"].get_bouquet_meanings(names)
    recommendations_payload = modules["recommendations"].get_bouquet_recommendations(names)
    return {
        "scientific_names": names,
        "meanings": meanings_payload.get("meanings", []),
        "recommendations": recommendations_payload.get("recommendations", []),
    }


def warm_route_caches() -> None:
    _load_search_modules()
    _recommend_with_fallback_copy("love", 5, "svd")
    _cached_visualizer_flowers(96)


def register_routes(app):
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    @app.route("/api/config")
    def config():
        return jsonify(
            {
                "use_llm": USE_LLM,
            }
        )

    @app.route("/api/episodes")
    def episodes_search():
        text = request.args.get("title", "")
        return jsonify(json_search(text))

    @app.route("/api/recommendations")
    def recommendations():
        query = request.args.get("q", "")
        method = request.args.get("method", "svd")
        limit = request.args.get("limit", default=5, type=int)
        limit = max(1, min(limit, 20))
        if not query or not query.strip():
            return jsonify(_recommend_with_fallback_copy(query, limit, method))
        
        retrieval_query, exclude_terms, _, _ = _cached_retrieval_query(query)
        payload = _recommend_with_fallback_copy(retrieval_query, limit, method)
        payload = _apply_hard_filters(payload, exclude_terms)
        payload["query"] = query
        return jsonify(payload)

    @app.route("/api/rag-recommendations")
    def rag_recommendations():
        query = request.args.get("q", "")
        method = request.args.get("method", "svd")
        limit = request.args.get("limit", default=5, type=int)
        limit = max(1, min(limit, 20))
        if not query or not query.strip():
            return jsonify(_recommend_with_fallback_copy(query, limit, method))
        return jsonify(_rag_recommendations(query, limit, method))

    @app.route("/api/rag-overview")
    def rag_overview():
        query = request.args.get("q", "")
        method = request.args.get("method", "svd")
        limit = request.args.get("limit", default=5, type=int)
        limit = max(1, min(limit, 20))
        if not query or not query.strip():
            return jsonify({"query": query, "rag": None})
        return jsonify(_rag_overview(query, limit, method))

    @app.route("/api/rag-card-summary", methods=["POST"])
    def rag_card_summary():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query") or "").strip()
        suggestion = payload.get("suggestion") or {}
        if not query:
            return jsonify({"error": "query is required."}), 400
        if not isinstance(suggestion, dict) or not suggestion.get("name"):
            return jsonify({"error": "suggestion is required."}), 400
        return jsonify(_rag_card_summary(query, suggestion))

    @app.route("/api/visualizer-flowers")
    def visualizer():
        limit = request.args.get("limit", default=50, type=int)
        limit = max(1, min(limit, 128))
        return jsonify(copy.deepcopy(_cached_visualizer_flowers(limit)))

    @app.route("/api/flower-images/<path:filename>")
    def flower_image(filename):
        return send_from_directory(FLOWER_IMAGE_DIR, filename)

    @app.route("/api/visualizer-bouquet-insights", methods=["POST"])
    def visualizer_bouquet_insights():
        payload = request.get_json(silent=True) or {}
        scientific_names = payload.get("scientific_names", [])
        if not isinstance(scientific_names, list):
            return jsonify({"error": "scientific_names must be a list."}), 400

        cleaned_names = [
            scientific_name.strip()
            for scientific_name in scientific_names
            if isinstance(scientific_name, str) and scientific_name.strip()
        ]
        if not cleaned_names:
            return jsonify({
                "scientific_names": [],
                "meanings": [],
                "recommendations": [],
            })

        return jsonify(copy.deepcopy(_cached_visualizer_bouquet_insights(tuple(cleaned_names))))

    @app.route("/api/autocomplete")
    def autocomplete():
        query = request.args.get("q", "")
        modules = _load_search_modules()
        return jsonify(modules["autocomplete_queries"](query))

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
