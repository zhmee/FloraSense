
# """
# Routes: React app serving and episode search API.

# To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
# """
# #NEW VERSION EXPERIMENTAL
# import json
# import logging
# import os
# import re
# import time
# from functools import lru_cache
# from importlib.machinery import SourceFileLoader
# from importlib.util import module_from_spec, spec_from_loader
# from pathlib import Path
# from flask import send_from_directory, request, jsonify
# from utils import FLOWER_IMAGE_DIR
# from models import db, Episode, Review
# from concurrent.futures import ThreadPoolExecutor, as_completed


# # ── AI toggle ────────────────────────────────────────────────────────────────
# USE_LLM = True
# # USE_LLM = True
# # ─────────────────────────────────────────────────────────────────────────────

# logger = logging.getLogger(__name__)

# VISUALIZATION_DIR = Path(__file__).resolve().parent / "3d_visualization"


# def json_search(query):
#     if not query or not query.strip():
#         query = "Kardashian"
#     results = db.session.query(Episode, Review).join(
#         Review, Episode.id == Review.id
#     ).filter(
#         Episode.title.ilike(f'%{query}%')
#     ).all()
#     matches = []
#     for episode, review in results:
#         matches.append({
#             'title': episode.title,
#             'descr': episode.descr,
#             'imdb_rating': review.imdb_rating
#         })
#     return matches


# def _load_python_source(module_name, source_path):
#     loader = SourceFileLoader(module_name, str(source_path))
#     spec = spec_from_loader(module_name, loader)
#     if spec is None:
#         raise ImportError(f"Could not load module spec for {source_path}")
#     module = module_from_spec(spec)
#     loader.exec_module(module)
#     return module


# @lru_cache(maxsize=1)
# def _load_search_modules():
#     from flower_autocomplete import autocomplete_queries
#     from flower_recommender_prototype4 import recommend_flowers, visualizer_flowers
#     from flower_recommender_v3 import recommend_flowers_tfidf

#     return {
#         "recommend_flowers": recommend_flowers,
#         "recommend_flowers_tfidf": recommend_flowers_tfidf,
#         "visualizer_flowers": visualizer_flowers,
#         "autocomplete_queries": autocomplete_queries,
#     }


# def _normalize_recommendation_payload(payload, query: str) -> dict:
#     normalized = dict(payload or {})
#     normalized["query"] = normalized.get("query", query)
#     normalized["keywords_used"] = normalized.get("keywords_used", [])
#     normalized["query_latent_radar_chart"] = normalized.get("query_latent_radar_chart")
#     normalized["query_latent_radar_axes"] = normalized.get("query_latent_radar_axes", [])
#     normalized["keywords_used"] = [
#         {
#             **dict(item),
#             "explanation": dict(item).get("explanation", ""),
#             "explanation_source": dict(item).get("explanation_source", ""),
#         }
#         for item in normalized["keywords_used"]
#         if isinstance(item, dict)
#     ]

#     suggestions = []
#     for suggestion in normalized.get("suggestions", []) or []:
#         item = dict(suggestion)
#         item["matched_keywords"] = item.get("matched_keywords", [])
#         item["latent_radar_chart"] = item.get("latent_radar_chart")
#         item["latent_radar_axes"] = item.get("latent_radar_axes", [])
#         item["query_fit_explanation"] = item.get("query_fit_explanation", "")
#         item["query_fit_occasion_summary"] = item.get("query_fit_occasion_summary", "")
#         item["occasion_summary_source"] = item.get("occasion_summary_source", "")
#         suggestions.append(item)
#     normalized["suggestions"] = suggestions

#     if "score_scale" not in normalized:
#         scores = [
#             float(suggestion.get("score", 0) or 0)
#             for suggestion in suggestions
#         ]
#         normalized["score_scale"] = "unit" if scores and max(scores) <= 1.0 else "percent"

#     return normalized


# def _add_recommendation_explanations(payload: dict, use_llm: bool = False) -> dict:
#     try:
#         from flower_explanations import add_query_fit_explanations

#         return add_query_fit_explanations(payload, use_llm=use_llm)
#     except Exception:
#         logger.exception("Could not build query-aware recommendation explanations.")
#         return payload


# def _recommend_with_fallback(query: str, limit: int, method: str, use_llm_explanations: bool = False) -> dict:
#     modules = _load_search_modules()

#     if method == "tfidf":
#         payload = modules["recommend_flowers_tfidf"](query, limit=limit)
#         return _add_recommendation_explanations(
#             _normalize_recommendation_payload(payload, query),
#             use_llm=use_llm_explanations,
#         )

#     try:
#         payload = modules["recommend_flowers"](query, limit=limit)
#         return _add_recommendation_explanations(
#             _normalize_recommendation_payload(payload, query),
#             use_llm=use_llm_explanations,
#         )
#     except Exception:
#         logger.exception("SVD recommendations failed for query %r. Falling back to TF-IDF.", query)
#         payload = modules["recommend_flowers_tfidf"](query, limit=limit)
#         return _add_recommendation_explanations(
#             _normalize_recommendation_payload(payload, query),
#             use_llm=use_llm_explanations,
#         )


# def _llm_client():
#     api_key = os.getenv("SPARK_API_KEY")
#     if not api_key:
#         try:
#             from dotenv import load_dotenv

#             load_dotenv()
#         except Exception:
#             logger.debug("python-dotenv is unavailable while loading Spark API key.", exc_info=True)
#         api_key = os.getenv("SPARK_API_KEY")

#     if not api_key:
#         return None, "SPARK_API_KEY is not set."

#     try:
#         from infosci_spark_client import LLMClient
#     except Exception:
#         logger.exception("Could not import infosci_spark_client for RAG.")
#         return None, "The LLM client could not be loaded."

#     return LLMClient(api_key=api_key), ""


# def _extract_json_object(content: str) -> dict:
#     content = (content or "").strip()
#     if not content:
#         return {}

#     try:
#         parsed = json.loads(content)
#         return parsed if isinstance(parsed, dict) else {}
#     except json.JSONDecodeError:
#         pass

#     match = re.search(r"\{.*\}", content, flags=re.DOTALL)
#     if not match:
#         return {}

#     try:
#         parsed = json.loads(match.group(0))
#     except json.JSONDecodeError:
#         return {}
#     return parsed if isinstance(parsed, dict) else {}


# def _llm_text_response(client, messages: list[dict]) -> str:
#     try:
#         response = client.chat(messages, stream=False, show_thinking=False)
#     except TypeError:
#         response = client.chat(messages)
#     return (response or {}).get("content", "").strip()


# QUERY_STOPWORDS = {
#     "a",
#     "about",
#     "and",
#     "any",
#     "are",
#     "actually",
#     "because",
#     "been",
#     "basically",
#     "but",
#     "can",
#     "could",
#     "couldn",
#     "couldnt",
#     "do",
#     "does",
#     "doesn",
#     "doesnt",
#     "don",
#     "doni",
#     "dont",
#     "for",
#     "give",
#     "have",
#     "help",
#     "i",
#     "im",
#     "in",
#     "is",
#     "isn",
#     "isnt",
#     "it",
#     "just",
#     "kinda",
#     "kind",
#     "like",
#     "looking",
#     "me",
#     "maybe",
#     "need",
#     "needs",
#     "not",
#     "of",
#     "or",
#     "please",
#     "quite",
#     "really",
#     "say",
#     "show",
#     "sort",
#     "sorta",
#     "someone",
#     "something",
#     "that",
#     "the",
#     "them",
#     "they",
#     "this",
#     "to",
#     "too",
#     "very",
#     "want",
#     "wasn",
#     "wasnt",
#     "won",
#     "wont",
#     "with",
#     "wouldn",
#     "wouldnt",
#     "who",
# }
# MEANING_INTENT_TOKENS = {
#     "courage",
#     "friendship",
#     "gratitude",
#     "hope",
#     "love",
#     "meaning",
#     "means",
#     "remembrance",
#     "represent",
#     "represents",
#     "romantic",
#     "romance",
#     "strength",
#     "symbol",
#     "symbolic",
#     "symbolism",
#     "symbolize",
#     "symbolizes",
#     "thanks",
#     "thank",
# }
# NEGATIVE_LOVE_MEANING_PATTERNS = (
#     "decrease of love",
#     "fading love",
#     "lost love",
#     "love denied",
#     "rejected love",
#     "unrequited love",
#     "infidelity",
#     "betrayal",
# )


# def _query_has_symbolic_intent(query: str) -> bool:
#     query_tokens = set(re.findall(r"[a-z0-9]+", _rag_name_key(query)))
#     return bool(query_tokens & MEANING_INTENT_TOKENS)


# def _meaning_value_supports_query_term(term: str, value: str) -> bool:
#     normalized_term = _rag_name_key(term)
#     normalized_value = _rag_name_key(value)
#     if not normalized_term or not normalized_value:
#         return False
#     if any(pattern in normalized_term for pattern in NEGATIVE_LOVE_MEANING_PATTERNS):
#         return False
#     if normalized_term == "love" and any(pattern in normalized_value for pattern in NEGATIVE_LOVE_MEANING_PATTERNS):
#         return False
#     return normalized_term == normalized_value or normalized_term in normalized_value or normalized_value in normalized_term

# def _llm_retrieval_query(client, user_query: str) -> tuple[str, list[str], str, str]:
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "Turn the user's request into a short flower search query for an IR system. "
#                 "Keep only searchable words: color, occasion, relationship, flower meaning, "
#                 "maintenance level, and flower names. Convert vague intent into simple flower "
#                 "meaning words such as gratitude, love, sympathy, friendship, remembrance, "
#                 "rebirth, strength, courage, low maintenance, or high maintenance. "
#                 "Remove filler words such as really, very, actually, basically, just, maybe, "
#                 "kind of, sort of, I, me, want, need, something, and please. "

#                 "NEGATION RULE: "
#                 "If the user negates any attribute (color, flower name, occasion, trait): "
#                 "  - OMIT that attribute from retrieval_query entirely. "
#                 "  - Add it to the exclude_terms list as a lowercase string. "
#                 "  - Only include attributes the user explicitly wants. "
#                 "If the user negates a trait with a clear opposite (e.g. 'not high maintenance'): "
#                 "  - Replace it with its opposite in retrieval_query (e.g. 'low maintenance'). "
#                 "  - Do NOT add it to exclude_terms in that case. "
#                 "Never include negation words (not, no, don't, without) in retrieval_query. "

#                 "Do not infer symbolic meaning from color constraints alone. "
#                 "Do not answer the user. Return JSON only: "
#                 "{\"retrieval_query\":\"short search phrase\","
#                 "\"exclude_terms\":[\"negated attributes as lowercase strings, or empty array\"],"
#                 "\"rationale\":\"brief reason\"}."
#             ),
#         },
#         {
#             "role": "user",
#             "content": (
#                 "Examples:\n"
#                 "User: I need flowers to say thanks to my teacher\n"
#                 "JSON: {\"retrieval_query\":\"gratitude teacher flowers\",\"exclude_terms\":[],\"rationale\":\"thanks maps to gratitude\"}\n"
#                 "User: something romantic but not too hard to take care of\n"
#                 "JSON: {\"retrieval_query\":\"love low maintenance flowers\",\"exclude_terms\":[],\"rationale\":\"not high maintenance flipped to low maintenance\"}\n"
#                 "User: I want flowers but not yellow and not roses\n"
#                 "JSON: {\"retrieval_query\":\"flowers\",\"exclude_terms\":[\"yellow\",\"rose\"],\"rationale\":\"yellow and rose excluded per user request\"}\n\n"
#                 f"User: {user_query}"
#             ),
#         },
#     ]

#     try:
#         content = _llm_text_response(client, messages)
#     except Exception:
#         logger.exception("LLM query transformation failed.")
#         return user_query, [], "LLM query transformation failed; using the original query.", "local"

#     parsed = _extract_json_object(content)
#     retrieval_query = str(parsed.get("retrieval_query") or "").strip()
#     rationale = str(parsed.get("rationale") or "").strip()
#     exclude_terms = [str(t).lower().strip() for t in parsed.get("exclude_terms") or [] if t]

#     if not retrieval_query:
#         return user_query, [], "The LLM did not return a valid retrieval query, so the original query was used.", "local"

#     return retrieval_query, exclude_terms, rationale, "llm"

# import re

# def _apply_hard_filters(payload: dict, exclude_terms: list[str]) -> dict:
#     if not exclude_terms:
#         return payload

#     filtered = []
#     for suggestion in payload.get("suggestions", []):
#         haystack = " ".join(
#             suggestion.get("colors", [])
#             + suggestion.get("meanings", [])
#             + suggestion.get("occasions", [])
#             + suggestion.get("plant_types", [])
#             + suggestion.get("maintenance", [])
#             + [suggestion.get("name", "")]
#         ).lower()

#         excluded = any(
#             re.search(rf"\b{re.escape(term)}\b", haystack)
#             for term in exclude_terms
#         )
#         if excluded:
#             continue
#         filtered.append(suggestion)

#     if not filtered:
#         logger.warning(f"Initial suggestions: {len(payload.get('suggestions', []))}")
#         return payload

#     payload["suggestions"] = filtered
#     return payload

# def _compact_context_values(values, limit: int = 3) -> list[str]:
#     if not isinstance(values, list):
#         return []

#     compacted = []
#     seen = set()
#     for value in values:
#         text = " ".join(str(value or "").split())
#         if not text:
#             continue
#         key = text.lower()
#         if key in seen:
#             continue
#         seen.add(key)
#         compacted.append(text)
#         if len(compacted) >= limit:
#             break
#     return compacted


# def _rag_name_key(value: str) -> str:
#     return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


# def _build_rag_context_documents(payload: dict, limit: int = 5) -> list[dict]:
#     documents = []
#     for rank, suggestion in enumerate((payload.get("suggestions") or [])[:limit], start=1):
#         matched_keywords = [
#             {
#                 "keyword": match.get("keyword", ""),
#                 "category": match.get("category", ""),
#             }
#             for match in (suggestion.get("matched_keywords") or [])[:8]
#             if isinstance(match, dict)
#         ]
#         documents.append(
#             {
#                 "rank": rank,
#                 "name": suggestion.get("name", ""),
#                 "scientific_name": suggestion.get("scientific_name", ""),
#                 "score": suggestion.get("score"),
#                 "non_rag_explanation": suggestion.get("query_fit_explanation", ""),
#                 "non_rag_occasion_summary": suggestion.get("query_fit_occasion_summary", ""),
#                 "has_occasion_evidence": bool(
#                     suggestion.get("query_fit_occasion_summary")
#                     or suggestion.get("occasions")
#                 ),
#                 "colors": _compact_context_values(suggestion.get("colors"), 5),
#                 "maintenance": _compact_context_values(suggestion.get("maintenance"), 2),
#                 "plant_types": _compact_context_values(suggestion.get("plant_types"), 3),
#                 "meanings": _compact_context_values(suggestion.get("meanings"), 5),
#                 "occasions": _compact_context_values(suggestion.get("occasions"), 5),
#                 "matched_keywords": matched_keywords,
#             }
#         )
#     return documents


# def _format_context_list(label: str, values: list[str]) -> str:
#     if not values:
#         return f"- **{label}:** Not listed"
#     return f"- **{label}:** {'; '.join(values)}"


# def _format_flower_rag_context(context_documents: list[dict]) -> str:
#     """Render retrieved flower records as Markdown context for the LLM.

#     This mirrors the in-class RAG demo shape: retrieval happens first, then the
#     full retrieved records are pasted into the final prompt as readable context.
#     """
#     parts = []
#     for document in context_documents:
#         matches = []
#         for match in document.get("matched_keywords", []) or []:
#             if not isinstance(match, dict):
#                 continue
#             keyword = " ".join(str(match.get("keyword") or "").split())
#             category = " ".join(str(match.get("category") or "").split())
#             if keyword and category:
#                 matches.append(f"{keyword} ({category})")
#             elif keyword:
#                 matches.append(keyword)

#         lines = [
#             f"### [{document.get('rank')}] {document.get('name') or 'Unnamed flower'}",
#             f"- **Score:** {document.get('score')}",
#             f"- **Scientific name:** {document.get('scientific_name') or 'Not listed'}",
#             f"- **Has occasion evidence:** {'yes' if document.get('has_occasion_evidence') else 'no'}",
#             _format_context_list("Colors", document.get("colors") or []),
#             _format_context_list("Plant types", document.get("plant_types") or []),
#             _format_context_list("Maintenance", document.get("maintenance") or []),
#             _format_context_list("Matched query evidence", matches),
#         ]

#         non_rag_explanation = " ".join(
#             str(document.get("non_rag_explanation") or "").split()
#         )
#         if non_rag_explanation:
#             lines.extend(
#                 [
#                     "",
#                     '**Existing "Why this matches" card text to improve:**',
#                     non_rag_explanation,
#                 ]
#             )

#         non_rag_occasion = " ".join(
#             str(document.get("non_rag_occasion_summary") or "").split()
#         )
#         if non_rag_occasion:
#             lines.extend(
#                 [
#                     "",
#                     "**Existing occasion fit text to improve:**",
#                     non_rag_occasion,
#                 ]
#             )

#         meanings = document.get("meanings") or []
#         if meanings:
#             lines.extend(["", "**Meanings:**"])
#             lines.extend(f"- {meaning}" for meaning in meanings)

#         occasions = document.get("occasions") or []
#         if occasions:
#             lines.extend(["", "**Occasions:**"])
#             lines.extend(f"- {occasion}" for occasion in occasions)

#         parts.append("\n".join(lines))

#     return "\n\n---\n\n".join(parts)


# def _local_card_summary(user_query: str, document: dict) -> str:
#     name = document.get("name") or "This flower"
#     matched_by_category: dict[str, list[str]] = {}
#     for match in document.get("matched_keywords", []) or []:
#         if not isinstance(match, dict):
#             continue
#         keyword = (match.get("keyword") or "").strip()
#         category = (match.get("category") or "semantic").strip()
#         if not keyword:
#             continue
#         matched_by_category.setdefault(category, [])
#         if keyword not in matched_by_category[category]:
#             matched_by_category[category].append(keyword)

#     requested_tokens = [
#         token
#         for token in re.findall(r"[a-z0-9]+", _rag_name_key(user_query))
#         if len(token) >= 4 and token not in QUERY_STOPWORDS and token not in {"flower", "flowers"}
#     ]
#     values_by_field = {
#         "color": document.get("colors", []) or [],
#         "meaning": document.get("meanings", []) or [],
#         "occasion": document.get("occasions", []) or [],
#         "maintenance": document.get("maintenance", []) or [],
#         "plant type": document.get("plant_types", []) or [],
#     }
#     supported = []
#     unsupported = []
#     for token in requested_tokens:
#         if any(token in _rag_name_key(value) for values in values_by_field.values() for value in values):
#             supported.append(token)
#         else:
#             unsupported.append(token)

#     reasons = []
#     color_terms = matched_by_category.get("color") or [
#         value for value in document.get("colors", []) if _rag_name_key(value) in requested_tokens
#     ]
#     raw_meaning_terms = matched_by_category.get("meaning", []) if _query_has_symbolic_intent(user_query) else []
#     meaning_terms = [
#         term
#         for term in raw_meaning_terms
#         if any(_meaning_value_supports_query_term(term, value) for value in document.get("meanings", []) or [])
#     ]
#     occasion_terms = matched_by_category.get("occasion", [])
#     maintenance_terms = matched_by_category.get("maintenance", [])
#     plant_terms = matched_by_category.get("plant_type", [])

#     if color_terms:
#         reasons.append(f"is available in {', '.join(color_terms[:2])}")
#     if meaning_terms:
#         reasons.append(f"supports the symbolism of {', '.join(meaning_terms[:2])}")
#     if occasion_terms:
#         reasons.append(f"fits {', '.join(occasion_terms[:2])}")
#     if maintenance_terms:
#         reasons.append(f"matches {', '.join(maintenance_terms[:1])} care")
#     if plant_terms:
#         reasons.append(f"is a {', '.join(plant_terms[:1])}")

#     if unsupported and reasons:
#         return (
#             f"{name} is a partial fit for \"{user_query}\": it {', and '.join(reasons[:2])}, "
#             f"but the visible evidence does not clearly support {', '.join(unsupported[:2])}."
#         )
#     if reasons:
#         return (
#             f"{name} fits \"{user_query}\" because it {', and '.join(reasons[:3])}."
#         )

#     existing_explanation = (document.get("non_rag_explanation") or "").strip()
#     if existing_explanation:
#         return existing_explanation

#     return f"{name} has limited support for \"{user_query}\" because the retrieved record has sparse descriptive evidence."


# def _local_data_summary(document: dict) -> str:
#     meanings = _compact_context_values(document.get("meanings"), 2)
#     occasions = _compact_context_values(document.get("occasions"), 2)
#     parts = []
#     if meanings:
#         parts.append(f"Meanings: {'; '.join(meanings)}")
#     if occasions:
#         parts.append(f"Occasions: {'; '.join(occasions)}")
#     if parts:
#         return " ".join(parts)

#     name = document.get("name") or "This flower"
#     return f"{name} does not have meaning or occasion text listed in the flower data."


# def _fallback_card_summaries(user_query: str, context_documents: list[dict]) -> dict[str, dict[str, str]]:
#     summaries = {}
#     for document in context_documents:
#         name = document.get("name")
#         if not name:
#             continue
#         data_summary = _local_data_summary(document)
#         summaries[_rag_name_key(name)] = {
#             "rag_summary": data_summary,
#             "ir_summary": data_summary,
#             "rag_occasion_summary": "",
#         }
#     return summaries


# def _join_overview_values(values: list[str]) -> str:
#     values = [value for value in values if value]
#     if not values:
#         return ""
#     if len(values) == 1:
#         return values[0]
#     if len(values) == 2:
#         return f"{values[0]} and {values[1]}"
#     return f"{', '.join(values[:-1])}, and {values[-1]}"


# def _overview_reason_phrase(document: dict, user_query: str) -> str:
#     matched_by_category: dict[str, list[str]] = {}
#     for match in document.get("matched_keywords", []) or []:
#         if not isinstance(match, dict):
#             continue
#         keyword = " ".join(str(match.get("keyword") or "").split())
#         category = " ".join(str(match.get("category") or "").split())
#         if keyword and category:
#             matched_by_category.setdefault(category, [])
#             if keyword not in matched_by_category[category]:
#                 matched_by_category[category].append(keyword)

#     reasons = []
#     meaning_terms = [
#         term
#         for term in (matched_by_category.get("meaning") or [])
#         if any(_meaning_value_supports_query_term(term, value) for value in document.get("meanings", []) or [])
#     ]
#     if _query_has_symbolic_intent(user_query) and meaning_terms:
#         reasons.append(f"its symbolism centers on {_join_overview_values(meaning_terms[:2])}")
#     if matched_by_category.get("occasion"):
#         reasons.append(f"it naturally suits {_join_overview_values(matched_by_category['occasion'][:2])}")
#     if matched_by_category.get("color"):
#         reasons.append(f"it is available in {_join_overview_values(matched_by_category['color'][:2])}")
#     if matched_by_category.get("maintenance"):
#         reasons.append(f"it matches {_join_overview_values(matched_by_category['maintenance'][:1])} care")
#     if matched_by_category.get("plant_type"):
#         reasons.append(f"it is a {_join_overview_values(matched_by_category['plant_type'][:1])}")

#     if reasons:
#         return _join_overview_values(reasons[:3])

#     meanings = _compact_context_values(document.get("meanings"), 2) if _query_has_symbolic_intent(user_query) else []
#     occasions = _compact_context_values(document.get("occasions"), 1)
#     if meanings and occasions:
#         return f"it offers {_join_overview_values(meanings)} symbolism and fits {occasions[0]}"
#     if meanings:
#         return f"its listed meaning emphasizes {_join_overview_values(meanings)}"
#     if occasions:
#         return f"it is tied to {occasions[0]}"
#     return "the retrieved flower details give it the strongest overall support"


# def _fallback_rag_answer(user_query: str, context_documents: list[dict]) -> str:
#     if not context_documents:
#         return "No matching flowers were found."

#     top_documents = [doc for doc in context_documents[:2] if doc.get("name")]
#     if not top_documents:
#         return "The strongest flower details are shown below."

#     first = top_documents[0]
#     first_name = first.get("name")
#     first_reason = _overview_reason_phrase(first, user_query)
#     if len(top_documents) == 1:
#         return f"For \"{user_query}\", {first_name} is the clearest fit because {first_reason}."

#     second = top_documents[1]
#     second_name = second.get("name")
#     second_reason = _overview_reason_phrase(second, user_query)
#     return (
#         f"For \"{user_query}\", {first_name} is the most direct fit because {first_reason}. "
#         f"{second_name} is worth considering when you want a slightly different emphasis: {second_reason}."
#     )


# def _generate_rag_response(
#     client,
#     user_query: str,
#     retrieval_query: str,
#     context_documents: list[dict],
# ) -> tuple[str, dict[str, dict[str, str]]]:
#     if not context_documents:
#         return "I could not find matching flower records to ground an answer.", {}

#     context_markdown = _format_flower_rag_context(context_documents)
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are the final reader in a retrieve-then-read flower recommendation system. "
#                 "Answer only from the retrieved flower records below. If a retrieved record only "
#                 "partly supports the user's request, say it is a partial fit instead of forcing a "
#                 "match. Do not invent meanings, colors, occasions, or care details. "
#                 "Write warm, polished florist-style prose, not a comma-separated inventory. "
#                 "The answer should compare the strongest flowers: explain what each is especially "
#                 "good for, and why someone might choose one over another. "
#                 "For every card summary, use the existing 'Why this matches' card text and "
#                 "occasion fit text as starting evidence, then rewrite both into one deeper "
#                 "user-facing explanation grounded in the retrieved record. Include why the flower "
#                 "matches the query, what evidence supports that match, and how its occasion fit "
#                 "matters when occasion evidence is present. If the existing text is awkward or "
#                 "thin, improve it with the listed colors, plant type, maintenance, meanings, "
#                 "occasions, and matched evidence. Do not copy weak fallback phrases like "
#                 "'fits mother day', 'visible evidence', or 'symbolism supports' verbatim. "
#                 "For every card occasion summary, rewrite the existing occasion fit text into "
#                 "a graceful sentence about when the flower is suitable. If a record says it has "
#                 "occasion evidence, rag_occasion_summary must not be empty; return an empty "
#                 "string only when the record has no occasion evidence. "
#                 "For every card ir_summary, summarize the retreived data with grammar improvements, "
#                 "so that it is more grammatically correct and easier to read. Keep most of the words from "
#                 "the original data source intact. ONLY polish it grammatically"
#                 "Never mention RAG, IR,"
#                 "vectors, retrieval, database, matched keywords, score, or context. "
#                 "Return JSON only with this exact shape: "
#                 "{\"answer\":\"2 to 3 sentence overall answer\", "
#                 "\"cards\":[{\"rank\":1,\"name\":\"exact Flower name\","
#                 "\"scientific_name\":\"exact scientific name\","
#                 "\"ir_summary\":\"1 to 2 concise evidence sentences for the raw card\","
#                 "\"rag_summary\":\"2 to 3 sentences grounded in the retrieved record\","
#                 "\"rag_occasion_summary\":\"one sentence about occasion fit or empty string\"}]}. "
#                 "The answer should be 2 to 3 graceful sentences, around 55 to 95 words total. "
#                 "The cards array must contain exactly one entry for every retrieved flower record, "
#                 "in the same order, using the exact rank, name, and scientific name shown. "
#                 "Do not skip duplicated or similar-looking records; each retrieved record needs "
#                 "its own ir_summary, rag_summary, and rag_occasion_summary. "
#                 "Each ir_summary must be around 5 useful sentences and must "
#                 "not copy the rag_summary. It must use directly the data for that particular flower, "
#                 "and just rewrite in a more user friendly readable way."
#                 "Each rag_summary must be 2 to 3 useful sentences, 45 to 80 words total, and "
#                 "should explain the fit using evidence that matters for the original user query."
#                 "Include WHY it was chosen. Each "
#                 "rag_occasion_summary must be one useful sentence, 14 to 30 words, focused only "
#                 "on occasion evidence."
#             ),
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"Original user query:\n{user_query}\n\n"
#                 f"LLM-transformed query sent to the flower IR system:\n{retrieval_query}\n\n"
#                 f"Retrieved flower records:\n\n{context_markdown}"
#             ),
#         },
#     ]

#     try:
#         content = _llm_text_response(client, messages)
#     except Exception:
#         logger.exception("LLM RAG answer generation failed.")
#         return "", {}

#     parsed = _extract_json_object(content)
#     answer = str(parsed.get("answer") or "").strip()
#     card_summaries: dict[str, dict[str, str]] = {}
#     cards = parsed.get("cards")
#     if isinstance(cards, list):
#         for card in cards:
#             if not isinstance(card, dict):
#                 continue
#             rank = card.get("rank")
#             try:
#                 rank_number = int(rank)
#             except (TypeError, ValueError):
#                 rank_number = None
#             name = str(card.get("name") or "").strip()
#             scientific_name = str(card.get("scientific_name") or "").strip()
#             ir_summary = str(card.get("ir_summary") or "").strip()
#             summary = str(card.get("rag_summary") or "").strip()
#             occasion_summary = str(card.get("rag_occasion_summary") or "").strip()
#             card_text = {
#                 "ir_summary": ir_summary,
#                 "rag_summary": summary,
#                 "rag_occasion_summary": occasion_summary,
#             }
#             if (ir_summary or summary or occasion_summary) and rank_number is not None:
#                 card_summaries[f"rank:{rank_number}"] = card_text
#             if name and (ir_summary or summary or occasion_summary):
#                 card_summaries[_rag_name_key(name)] = card_text
#             if scientific_name and (ir_summary or summary or occasion_summary):
#                 card_summaries[f"scientific:{_rag_name_key(scientific_name)}"] = card_text

#     if not answer:
#         answer = content

#     return answer, card_summaries
# def _generate_rag_overview_only(client, user_query, retrieval_query, context_documents) -> str:
#     if not context_documents:
#         return _fallback_rag_answer(user_query, context_documents)

#     context_markdown = _format_flower_rag_context(context_documents)
#     messages = [
#         {
#             "role": "system",
#             "content": (
#             """
#             You are the final reader in a retrieve-then-read flower recommendation system. 
#             Answer only from the retrieved flower records below. If a retrieved record only 
#             partly supports the user's request, say it is a partial fit instead of forcing a 
#             match. Do not invent meanings, colors, occasions, or care details. 

#             Write warm, polished florist-style prose, not a comma-separated inventory. 
#             The answer should compare the strongest flowers: explain what each is especially 
#             good for, and why someone might choose one over another. 

#             Never mention RAG, IR, vectors, retrieval, database, matched keywords, score, or context. 

#             Return JSON only with this exact shape: 
#             {"answer":"2 to 3 sentence overall answer"}. 

#             The answer should be 2 to 3 graceful sentences, around 55 to 95 words total.
#             """
#             ),
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"Original user query:\n{user_query}\n\n"
#                 f"LLM-transformed query sent to the flower IR system:\n{retrieval_query}\n\n"
#                 f"Retrieved flower records:\n\n{context_markdown}"
#             ),
#         },
#     ]
#     try:
#         content = _llm_text_response(client, messages).strip()
#         parsed = _extract_json_object(content)
#         return str(parsed.get("answer") or "").strip() or content
#     except Exception:
#         logger.exception("LLM overview generation failed.")
#         return _fallback_rag_answer(user_query, context_documents)


# def _generate_rag_cards_only(client, user_query, retrieval_query, context_documents) -> dict[str, dict[str, str]]:
#     if not context_documents:
#         return {}

#     context_markdown = _format_flower_rag_context(context_documents)
#     messages = [
#         {
#             "role": "system",
#             "content": (
#              """
#             You are the final reader in a retrieve-then-read flower recommendation system. 
#             Answer only from the retrieved flower records below. Do not invent meanings, colors, occasions, or care details.

#             For every card summary, use the existing 'Why this matches' card text and 
#             occasion fit text as starting evidence, then rewrite both into one deeper 
#             user-facing explanation (rag_summary) grounded in the retrieved record. Include why the flower 
#             matches the query, what evidence supports that match, and how its occasion fit 
#             matters when occasion evidence is present. If the existing text is awkward or 
#             thin, improve it with the listed colors, plant type, maintenance, meanings, 
#             occasions, and matched evidence. Do not copy weak fallback phrases verbatim.

#             For every card ir_summary, summarize the retrieved data with grammar improvements 
#             so that it is more grammatically correct and easier to read. Keep most of the words 
#             from the original data source intact. ONLY polish it grammatically.

#             For every card occasion summary, rewrite the existing occasion fit text into 
#             a graceful sentence. If a record says it has occasion evidence, rag_occasion_summary 
#             must not be empty; return an empty string only when the record has no occasion evidence.

#             Never mention RAG, IR, vectors, retrieval, database, matched keywords, score, or context. 

#             Return JSON only with this exact shape: 
#             {
#             "cards":[
#                 {
#                 "rank": 1,
#                 "name": "exact Flower name",
#                 "scientific_name": "exact scientific name",
#                 "ir_summary": "around 5 useful sentences",
#                 "rag_summary": "2 to 3 sentences, 45 to 80 words total",
#                 "rag_occasion_summary": "one sentence, 14 to 30 words"
#                 }
#             ]
#             }

#             The cards array must contain exactly one entry for every retrieved flower record, 
#             in the same order. Each ir_summary must not copy the rag_summary. 
#             Each rag_summary should explain the fit using evidence that matters for the original 
#             user query and include WHY it was chosen.
#             """
#             ),
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"Original user query:\n{user_query}\n\n"
#                 f"LLM-transformed query sent to the flower IR system:\n{retrieval_query}\n\n"
#                 f"Retrieved flower records:\n\n{context_markdown}"
#             ),
#         },
#     ]
#     try:
#         content = _llm_text_response(client, messages)
#     except Exception:
#         logger.exception("LLM card generation failed.")
#         return {}

#     parsed = _extract_json_object(content)
#     card_summaries: dict[str, dict[str, str]] = {}
#     for card in parsed.get("cards") or []:
#         if not isinstance(card, dict):
#             continue
#         rank = card.get("rank")
#         name = str(card.get("name") or "").strip()
#         scientific_name = str(card.get("scientific_name") or "").strip()
#         card_text = {
#             "ir_summary": str(card.get("ir_summary") or "").strip(),
#             "rag_summary": str(card.get("rag_summary") or "").strip(),
#             "rag_occasion_summary": str(card.get("rag_occasion_summary") or "").strip(),
#         }
#         try:
#             card_summaries[f"rank:{int(rank)}"] = card_text
#         except (TypeError, ValueError):
#             pass
#         if name:
#             card_summaries[_rag_name_key(name)] = card_text
#         if scientific_name:
#             card_summaries[f"scientific:{_rag_name_key(scientific_name)}"] = card_text

#     return card_summaries

# @lru_cache(maxsize=256)
# def _cached_retrieval_query(query: str):
#     client, reason = _llm_client()
#     if client is None:
#         return query, [], reason, "local"
#     return _llm_retrieval_query(client, query)

# def _rag_recommendations(query: str, limit: int, method: str, phase: str = "full") -> dict:
#     retrieval_query, exclude_terms, transform_rationale, transform_source = _cached_retrieval_query(query)

#     payload = _recommend_with_fallback(retrieval_query, limit * 2, method, use_llm_explanations=False)
#     payload = _apply_hard_filters(payload, exclude_terms)
#     payload["suggestions"] = payload["suggestions"][:limit]  # trim instead of re-fetching
#     payload["query"] = query
#     context_documents = _build_rag_context_documents(payload, limit=limit)

#     client, unavailable_reason = _llm_client()
#     answer = ""
#     card_summaries = {}

#     if client is not None:
#         if phase == "ir":
#             # Only need overview answer, skip cards
#             answer = _generate_rag_overview_only(client, query, retrieval_query, context_documents)
#         else:
#             # Run overview + cards concurrently — they don't depend on each other!
#             with ThreadPoolExecutor() as ex:
#                 overview_future = ex.submit(
#                     _generate_rag_overview_only, client, query, retrieval_query, context_documents
#                 )
#                 cards_future = ex.submit(
#                     _generate_rag_cards_only, client, query, retrieval_query, context_documents
#                 )
#                 answer = overview_future.result()
#                 card_summaries = cards_future.result()

#     if not answer:
#         answer = _fallback_rag_answer(query, context_documents)
#     if not card_summaries:
#         card_summaries = _fallback_card_summaries(query, context_documents)

#     # Attach card text to suggestions
#     for i, suggestion in enumerate(payload.get("suggestions", []), start=1):
#         name = suggestion.get("name", "")
#         scientific_name = suggestion.get("scientific_name", "")
#         card_text = (
#             card_summaries.get(f"rank:{i}")
#             or card_summaries.get(_rag_name_key(name))
#             or card_summaries.get(f"scientific:{_rag_name_key(scientific_name)}")
#             or {}
#         )
#         data_summary = _local_data_summary({
#             "name": name,
#             "meanings": suggestion.get("meanings", []),
#             "occasions": suggestion.get("occasions", []),
#         })
#         suggestion["ir_summary"] = card_text.get("ir_summary") or data_summary
#         suggestion["ir_summary_source"] = "llm" if card_text.get("ir_summary") else "local"
#         suggestion["rag_summary"] = card_text.get("rag_summary") or data_summary
#         suggestion["rag_occasion_summary"] = card_text.get("rag_occasion_summary") or suggestion.get("query_fit_occasion_summary", "")
#         suggestion["rag_source"] = "llm" if card_text.get("rag_summary") else "local"
#         suggestion["ir_query_fit_explanation"] = suggestion.get("query_fit_explanation", "")

#     payload["rag"] = {
#         "user_query": query,
#         "retrieval_query": retrieval_query,
#         "query_transform_source": transform_source,
#         "query_transform_rationale": transform_rationale,
#         "answer": answer,
#         "answer_source": "llm" if answer else "local",
#         "context_documents": context_documents,
#     }
#     return payload




# @lru_cache(maxsize=1)
# def _load_visualizer_insight_modules():
#     return {
#         "health": _load_python_source(
#             "visualizer_health_bar_calculation",
#             VISUALIZATION_DIR / "health_bar_calculation",
#         ),
#         "recommendations": _load_python_source(
#             "visualizer_recommendation_calculation",
#             VISUALIZATION_DIR / "recommendation_calculation",
#         ),
#     }

# def register_routes(app):
#     @app.route('/', defaults={'path': ''})
#     @app.route('/<path:path>')
#     def serve(path):
#         if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
#             return send_from_directory(app.static_folder, path)
#         else:
#             return send_from_directory(app.static_folder, 'index.html')

#     @app.route("/api/config")
#     def config():
#         return jsonify(
#             {
#                 "use_llm": USE_LLM,
#             }
#         )

#     @app.route("/api/episodes")
#     def episodes_search():
#         text = request.args.get("title", "")
#         return jsonify(json_search(text))

#     @app.route("/api/recommendations")
#     def recommendations():
#         query = request.args.get("q", "")
#         method = request.args.get("method", "svd") # SVD or TF-IDF # TODO: IMPLEMENT 
#         limit = request.args.get("limit", default=5, type=int)
#         limit = max(1, min(limit, 20))
#         return jsonify(_recommend_with_fallback(query, limit, method))

#     # In register_routes, update the rag-recommendations handler:
#     @app.route("/api/rag-recommendations")
#     def rag_recommendations():
#         query = request.args.get("q", "")
#         method = request.args.get("method", "svd")
#         limit = request.args.get("limit", default=5, type=int)
#         phase = request.args.get("phase", "full")  # "ir" | "overview" | "cards" | "full"
#         limit = max(1, min(limit, 20))
#         if not query or not query.strip():
#             return jsonify(_recommend_with_fallback(query, limit, method))
#         return jsonify(_rag_recommendations(query, limit, method, phase=phase))

#     @app.route("/api/visualizer-flowers")
#     def visualizer():
#         limit = request.args.get("limit", default=48, type=int)
#         modules = _load_search_modules()
#         return jsonify(modules["visualizer_flowers"](limit=limit))

#     @app.route("/api/flower-images/<path:filename>")
#     def flower_image(filename):
#         return send_from_directory(FLOWER_IMAGE_DIR, filename)

#     @app.route("/api/visualizer-bouquet-insights", methods=["POST"])
#     def visualizer_bouquet_insights():
#         payload = request.get_json(silent=True) or {}
#         scientific_names = payload.get("scientific_names", [])
#         if not isinstance(scientific_names, list):
#             return jsonify({"error": "scientific_names must be a list."}), 400

#         cleaned_names = [
#             scientific_name.strip()
#             for scientific_name in scientific_names
#             if isinstance(scientific_name, str) and scientific_name.strip()
#         ]
#         if not cleaned_names:
#             return jsonify({
#                 "scientific_names": [],
#                 "meanings": [],
#                 "recommendations": [],
#             })

#         modules = _load_visualizer_insight_modules()
#         meanings_payload = modules["health"].get_bouquet_meanings(cleaned_names)
#         recommendations_payload = modules["recommendations"].get_bouquet_recommendations(cleaned_names)
#         return jsonify({
#             "scientific_names": cleaned_names,
#             "meanings": meanings_payload.get("meanings", []),
#             "recommendations": recommendations_payload.get("recommendations", []),
#         })

#     @app.route("/api/autocomplete")
#     def autocomplete():
#         query = request.args.get("q", "")
#         modules = _load_search_modules()
#         return jsonify(modules["autocomplete_queries"](query))


#     if USE_LLM:
#         from llm_routes import register_chat_route
#         register_chat_route(app, json_search)

