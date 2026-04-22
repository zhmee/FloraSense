"""
Routes: React app serving and episode search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import json
import logging
import os
import re
from functools import lru_cache
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from flask import send_from_directory, request, jsonify
from utils import FLOWER_IMAGE_DIR
from models import db, Episode, Review

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = False
# USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

VISUALIZATION_DIR = Path(__file__).resolve().parent / "3d_visualization"


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


def _add_recommendation_explanations(payload: dict) -> dict:
    try:
        from flower_explanations import add_query_fit_explanations

        return add_query_fit_explanations(payload)
    except Exception:
        logger.exception("Could not build query-aware recommendation explanations.")
        return payload


def _recommend_with_fallback(query: str, limit: int, method: str) -> dict:
    modules = _load_search_modules()

    if method == "tfidf":
        payload = modules["recommend_flowers_tfidf"](query, limit=limit)
        return _add_recommendation_explanations(_normalize_recommendation_payload(payload, query))

    try:
        payload = modules["recommend_flowers"](query, limit=limit)
        return _add_recommendation_explanations(_normalize_recommendation_payload(payload, query))
    except Exception:
        logger.exception("SVD recommendations failed for query %r. Falling back to TF-IDF.", query)
        payload = modules["recommend_flowers_tfidf"](query, limit=limit)
        return _add_recommendation_explanations(_normalize_recommendation_payload(payload, query))


def _llm_client():
    api_key = os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            logger.debug("python-dotenv is unavailable while loading Spark API key.", exc_info=True)
        api_key = os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")

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

def _llm_retrieval_query(client, user_query: str) -> tuple[str, str]:
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
                "Resolve negations into positive search terms: don't want hard flowers, "
                "not high maintenance, or don't need much care should become low maintenance flowers; "
                "don't want easy care or not low maintenance should become high maintenance flowers. "
                "Never include negation words like not, don't, dont, or don in the retrieval_query. "
                "Do not answer the user. Return JSON only: "
                "{\"retrieval_query\":\"short search phrase\", \"rationale\":\"brief reason\"}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Examples:\n"
                "User: I need flowers to say thanks to my teacher\n"
                "JSON: {\"retrieval_query\":\"gratitude teacher flowers\",\"rationale\":\"thanks maps to gratitude\"}\n"
                "User: something romantic but not too hard to take care of\n"
                "JSON: {\"retrieval_query\":\"love low maintenance flowers\",\"rationale\":\"romantic maps to love and easy care maps to low maintenance\"}\n\n"
                f"User: {user_query}"
            ),
        },
    ]

    try:
        content = _llm_text_response(client, messages)
    except Exception:
        logger.exception("LLM query transformation failed.")
        return user_query, "LLM query transformation failed; using the original query."

    parsed = _extract_json_object(content)
    retrieval_query = str(parsed.get("retrieval_query") or "").strip()
    rationale = str(parsed.get("rationale") or "").strip()
    if not retrieval_query:
        retrieval_query = user_query
        rationale = "The LLM did not return a valid retrieval query, so the original query was used."

    return retrieval_query, rationale


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
            _format_context_list("Colors", document.get("colors") or []),
            _format_context_list("Plant types", document.get("plant_types") or []),
            _format_context_list("Maintenance", document.get("maintenance") or []),
            _format_context_list("Matched query evidence", matches),
        ]

        non_rag_explanation = " ".join(
            str(document.get("non_rag_explanation") or "").split()
        )
        if non_rag_explanation:
            lines.extend(["", "**Non-RAG card text:**", non_rag_explanation])

        non_rag_occasion = " ".join(
            str(document.get("non_rag_occasion_summary") or "").split()
        )
        if non_rag_occasion:
            lines.extend(["", "**Non-RAG occasion text:**", non_rag_occasion])

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


def _local_card_summary(user_query: str, document: dict) -> str:
    name = document.get("name") or "This flower"
    matched_by_category: dict[str, list[str]] = {}
    for match in document.get("matched_keywords", []) or []:
        if not isinstance(match, dict):
            continue
        keyword = (match.get("keyword") or "").strip()
        category = (match.get("category") or "semantic").strip()
        if not keyword:
            continue
        matched_by_category.setdefault(category, [])
        if keyword not in matched_by_category[category]:
            matched_by_category[category].append(keyword)

    requested_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", _rag_name_key(user_query))
        if len(token) >= 4 and token not in QUERY_STOPWORDS and token not in {"flower", "flowers"}
    ]
    values_by_field = {
        "color": document.get("colors", []) or [],
        "meaning": document.get("meanings", []) or [],
        "occasion": document.get("occasions", []) or [],
        "maintenance": document.get("maintenance", []) or [],
        "plant type": document.get("plant_types", []) or [],
    }
    supported = []
    unsupported = []
    for token in requested_tokens:
        if any(token in _rag_name_key(value) for values in values_by_field.values() for value in values):
            supported.append(token)
        else:
            unsupported.append(token)

    reasons = []
    color_terms = matched_by_category.get("color") or [
        value for value in document.get("colors", []) if _rag_name_key(value) in requested_tokens
    ]
    meaning_terms = matched_by_category.get("meaning", [])
    occasion_terms = matched_by_category.get("occasion", [])
    maintenance_terms = matched_by_category.get("maintenance", [])
    plant_terms = matched_by_category.get("plant_type", [])

    if color_terms:
        reasons.append(f"matches the requested {', '.join(color_terms[:2])} color")
    if meaning_terms:
        reasons.append(f"supports the symbolism of {', '.join(meaning_terms[:2])}")
    if occasion_terms:
        reasons.append(f"fits {', '.join(occasion_terms[:2])}")
    if maintenance_terms:
        reasons.append(f"matches {', '.join(maintenance_terms[:1])} care")
    if plant_terms:
        reasons.append(f"is a {', '.join(plant_terms[:1])}")

    if unsupported and reasons:
        return (
            f"{name} is a partial fit for \"{user_query}\": it {', and '.join(reasons[:2])}, "
            f"but the visible evidence does not clearly support {', '.join(unsupported[:2])}."
        )
    if reasons:
        return (
            f"{name} fits \"{user_query}\" because it {', and '.join(reasons[:3])}."
        )

    existing_explanation = (document.get("non_rag_explanation") or "").strip()
    if existing_explanation:
        return existing_explanation

    return f"{name} has limited support for \"{user_query}\" because the retrieved record has sparse descriptive evidence."


def _fallback_card_summaries(user_query: str, context_documents: list[dict]) -> dict[str, str]:
    return {
        _rag_name_key(document.get("name", "")): _local_card_summary(user_query, document)
        for document in context_documents
        if document.get("name")
    }


def _fallback_rag_answer(context_documents: list[dict], unavailable_reason: str) -> str:
    if not context_documents:
        return f"LLM answer unavailable: {unavailable_reason} No retrieved flowers were found."

    names = [doc["name"] for doc in context_documents[:3] if doc.get("name")]
    if not names:
        return f"LLM answer unavailable: {unavailable_reason} The retrieved context is shown below."
    if len(names) == 1:
        shortlist = names[0]
    elif len(names) == 2:
        shortlist = f"{names[0]} and {names[1]}"
    else:
        shortlist = f"{', '.join(names[:-1])}, and {names[-1]}"
    return f"LLM answer unavailable: {unavailable_reason} The IR system retrieved {shortlist} as the strongest context."


def _generate_rag_response(
    client,
    user_query: str,
    retrieval_query: str,
    context_documents: list[dict],
) -> tuple[str, dict[str, str]]:
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
                "Write natural card text, not a comma-separated inventory. Never mention RAG, IR, "
                "vectors, retrieval, database, matched keywords, score, or context. "
                "Return JSON only with this exact shape: "
                "{\"answer\":\"one sentence overall answer\", \"cards\":[{\"name\":\"Flower name\","
                "\"rag_summary\":\"one sentence grounded in the retrieved record\"}]}. "
                "Each rag_summary must be one useful sentence, 18 to 38 words, and should explain "
                "the fit using evidence that matters for the original user query."
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
    card_summaries = {}
    cards = parsed.get("cards")
    if isinstance(cards, list):
        for card in cards:
            if not isinstance(card, dict):
                continue
            name = str(card.get("name") or "").strip()
            summary = str(card.get("rag_summary") or "").strip()
            if name and summary:
                card_summaries[_rag_name_key(name)] = summary

    if not answer:
        answer = content

    return answer, card_summaries


def _rag_recommendations(query: str, limit: int, method: str) -> dict:
    client, unavailable_reason = _llm_client()
    if client is None:
        retrieval_query = query
        transform_rationale = f"{unavailable_reason} Using the original query because AI query rewriting is unavailable."
        transform_source = "local"
    else:
        retrieval_query, transform_rationale = _llm_retrieval_query(client, query)
        transform_source = "llm"

    payload = _recommend_with_fallback(retrieval_query, limit, method)
    payload["query"] = query
    context_documents = _build_rag_context_documents(payload, limit=limit)

    answer = ""
    card_summaries = {}
    answer_source = "llm"
    card_summary_source = "llm"
    if client is not None:
        answer, card_summaries = _generate_rag_response(
            client,
            query,
            retrieval_query,
            context_documents,
        )

    if not answer:
        answer_source = "local"
        answer = _fallback_rag_answer(
            context_documents,
            unavailable_reason or "the LLM response could not be generated.",
        )
    if not card_summaries:
        card_summaries = _fallback_card_summaries(query, context_documents)
        card_summary_source = "local"

    for suggestion in payload.get("suggestions", []) or []:
        name = suggestion.get("name", "")
        name_key = _rag_name_key(name)
        suggestion["rag_summary"] = card_summaries.get(name_key) or _local_card_summary(
            query,
            {
                "name": name,
                "non_rag_explanation": suggestion.get("query_fit_explanation", ""),
                "non_rag_occasion_summary": suggestion.get("query_fit_occasion_summary", ""),
                "matched_keywords": suggestion.get("matched_keywords", []),
                "meanings": suggestion.get("meanings", []),
                "occasions": suggestion.get("occasions", []),
                "colors": suggestion.get("colors", []),
                "maintenance": suggestion.get("maintenance", []),
                "plant_types": suggestion.get("plant_types", []),
            },
        )
        suggestion["rag_source"] = card_summary_source if name_key in card_summaries else "local"

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
        method = request.args.get("method", "svd") # SVD or TF-IDF # TODO: IMPLEMENT 
        limit = request.args.get("limit", default=5, type=int)
        limit = max(1, min(limit, 20))
        return jsonify(_recommend_with_fallback(query, limit, method))

    @app.route("/api/rag-recommendations")
    def rag_recommendations():
        query = request.args.get("q", "")
        method = request.args.get("method", "svd")
        limit = request.args.get("limit", default=5, type=int)
        limit = max(1, min(limit, 20))
        if not query or not query.strip():
            return jsonify(_recommend_with_fallback(query, limit, method))
        return jsonify(_rag_recommendations(query, limit, method))

    @app.route("/api/visualizer-flowers")
    def visualizer():
        limit = request.args.get("limit", default=48, type=int)
        modules = _load_search_modules()
        return jsonify(modules["visualizer_flowers"](limit=limit))

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

        modules = _load_visualizer_insight_modules()
        meanings_payload = modules["health"].get_bouquet_meanings(cleaned_names)
        recommendations_payload = modules["recommendations"].get_bouquet_recommendations(cleaned_names)
        return jsonify({
            "scientific_names": cleaned_names,
            "meanings": meanings_payload.get("meanings", []),
            "recommendations": recommendations_payload.get("recommendations", []),
        })

    @app.route("/api/autocomplete")
    def autocomplete():
        query = request.args.get("q", "")
        modules = _load_search_modules()
        return jsonify(modules["autocomplete_queries"](query))

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
