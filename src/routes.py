"""
Routes: React app serving and episode search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import json
import logging
import os
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
