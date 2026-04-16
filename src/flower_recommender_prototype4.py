"""
FloraSense prototype 4: semantic-first retrieval driven primarily by CSV data.

This version keeps the structured CSV meanings/occasions as the main signal and
uses a tighter LSA projection over those structured documents so semantic
similarity actually drives ranking instead of sitting beside lexical matching.
Scraped article text is used only as a small fallback when a flower is missing
enough structured meaning data.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import flower_recommender_prototype3 as p3
from flower_radar_chart import build_latent_radar_chart, select_latent_axes


MAX_SVD_COMPONENTS = 24
MIN_SVD_COMPONENTS = 12
MAX_QUERY_EXPANSIONS = 4
SEMANTIC_SIMILARITY_WEIGHT = 0.92
LEXICAL_STABILITY_WEIGHT = 0.08
FACET_SEMANTIC_WEIGHT = 0.88
FACET_LEXICAL_WEIGHT = 0.12
FACET_MIN_SCORE = 0.16
FACET_MIN_RELATIVE_SCORE = 0.58
FACET_CATEGORY_LIMITS = {
    "maintenance": 1,
    "occasion": 2,
    "meaning": 2,
}
FACET_CATEGORY_PRIORITY = {
    "maintenance": 0,
    "occasion": 1,
    "meaning": 2,
    "color": 3,
    "plant_type": 4,
}
MAINTENANCE_FACET_GLOSSES = {
    "low": (
        "easy to grow",
        "needs very little attention",
        "takes minimal upkeep",
    ),
    "medium": (
        "needs regular attention",
        "takes some upkeep",
        "requires moderate attention",
    ),
    "high": (
        "needs a lot of care",
        "takes a lot of care",
        "requires extra upkeep",
        "takes more effort to maintain",
        "is hard to maintain",
        "needs constant attention",
    ),
}
MAINTENANCE_HINT_TOKENS = {
    "attention",
    "care",
    "effort",
    "maintain",
    "maintenance",
    "upkeep",
}
SEMANTIC_BENCHMARK_QUERIES = (
    "a flower for secret love",
    "something for remembrance of a missing friend",
    "a flower for rebirth and enlightenment",
    "a flower that represents strength and courage",
    "hidden affection",
    "memory of a lost friend",
    "renewal and awakening",
    "inner strength and bravery",
    "gratitude",
    "loveliness",
)


def _meaning_chunks(flower: dict, limit: int = 12) -> list[str]:
    chunks = []
    for meaning in flower["meanings"]:
        chunks.extend(p3._split_meaning_chunks(meaning))
    return p3._dedupe_preserve_order(chunks)[:limit]


def _occasion_chunks(flower: dict, limit: int = 10) -> list[str]:
    chunks = []
    for occasion in flower["occasions"]:
        chunks.extend(p3._split_meaning_chunks(occasion))
    return p3._dedupe_preserve_order(chunks)[:limit]


def _build_semantic_passages(flower: dict) -> list[str]:
    name = flower["name"]
    passages = [f"{name} is a flower."]

    if flower["scientific_name"] != "Unknown":
        passages.append(f"{name} is also known as {flower['scientific_name']}.")

    meaning_chunks = _meaning_chunks(flower)
    occasion_chunks = _occasion_chunks(flower)

    for chunk in meaning_chunks:
        passages.append(f"{name} symbolizes {chunk}.")
        passages.append(f"{name} meaning {chunk}.")

    for occasion in occasion_chunks:
        passages.append(f"{name} is appropriate for {occasion}.")
        passages.append(f"{name} occasion {occasion}.")

    if flower["colors"]:
        passages.append(f"{name} flowers come in {p3._join_human(flower['colors'])}.")
    if flower["plant_types"]:
        passages.append(f"{name} is a {p3._join_human(flower['plant_types'])} plant.")
    for maintenance in flower["maintenance"]:
        passages.append(f"{name} is a {maintenance} maintenance flower.")
        for gloss in MAINTENANCE_FACET_GLOSSES.get(p3._normalize(maintenance), ()):
            passages.append(f"{name} flowers are {gloss}.")

    if not meaning_chunks and not occasion_chunks:
        passages.extend(flower["article_passages"][:1])
    elif not meaning_chunks and flower["article_passages"]:
        passages.extend(flower["article_passages"][:1])

    return p3._dedupe_preserve_order(passages)


def _build_document(flower: dict) -> tuple[list[str], str]:
    passages = _build_semantic_passages(flower)
    document_parts = [flower["name"]]
    if flower["scientific_name"] != "Unknown":
        document_parts.append(flower["scientific_name"])
    document_parts.extend(passages)
    return passages, "\n\n".join(part for part in document_parts if part)


def _keyword_in_query(keyword: str, normalized_query: str) -> bool:
    normalized_keyword = p3._normalize(keyword)
    if not normalized_keyword or not normalized_query:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(normalized_keyword)}(?!\w)", normalized_query))


def _facet_support_sentences(category: str, value: str, flower: dict) -> list[str]:
    name = flower["name"]
    if category == "meaning":
        return [
            f"{name} symbolizes {value}.",
            f"{name} meaning {value}.",
        ]
    if category == "occasion":
        return [
            f"{name} is appropriate for {value}.",
            f"{name} occasion {value}.",
        ]
    if category == "maintenance":
        display_value = p3._format_keyword_for_display(category, value)
        return [
            f"{name} is a {display_value} flower.",
            f"{name} has {display_value} needs.",
        ]
    if category == "color":
        return [
            f"{name} flowers come in {value}.",
            f"{value} flowers include {name}.",
        ]
    if category == "plant_type":
        return [
            f"{name} is a {value} plant.",
            f"{value} flowers include {name}.",
        ]
    return []


def _build_facet_document(category: str, value: str, supporting_flowers: list[dict]) -> str:
    display_keyword = p3._format_keyword_for_display(category, value)
    parts = [display_keyword, f"{display_keyword} {category}"]

    if category == "maintenance":
        for gloss in MAINTENANCE_FACET_GLOSSES.get(p3._normalize(value), ()):
            parts.append(f"{display_keyword} flowers are {gloss}.")
        return "\n".join(p3._dedupe_preserve_order(parts))

    for flower in supporting_flowers[:8]:
        parts.extend(_facet_support_sentences(category, value, flower))

    return "\n".join(p3._dedupe_preserve_order(parts))


def _facet_query_is_supported(category: str, query: str) -> bool:
    if category != "maintenance":
        return True

    query_tokens = set(p3._tokenize(query))
    if not query_tokens:
        return False
    return bool(query_tokens & MAINTENANCE_HINT_TOKENS)


def _extract_exact_structured_keywords(query: str, flowers: list[dict], limit: int) -> list[dict]:
    normalized_query = p3._normalize(query)
    if not normalized_query:
        return []

    candidates = []
    seen = set()
    for category, label in p3._collect_keyword_candidates(flowers=flowers):
        if category not in {"meaning", "occasion"}:
            continue

        keyword = p3._format_keyword_for_display(category, label)
        normalized_keyword = p3._normalize(keyword)
        if not normalized_keyword or (category, normalized_keyword) in seen:
            continue
        if not _keyword_in_query(keyword, normalized_query):
            continue

        seen.add((category, normalized_keyword))
        candidates.append(
            {
                "keyword": keyword,
                "category": category,
                "score": round(10 + len(p3._tokenize(keyword)) / 10, 2),
            }
        )

    candidates.sort(
        key=lambda item: (
            -len(p3._tokenize(item["keyword"])),
            FACET_CATEGORY_PRIORITY.get(item["category"], 99),
            item["keyword"],
        )
    )
    return candidates[:limit]


def _locked_facet_categories(query: str, query_keywords: list[dict]) -> set[str]:
    normalized_query = p3._normalize(query)
    present_keywords = [
        item
        for item in query_keywords
        if _keyword_in_query(item.get("keyword", ""), normalized_query)
    ]
    meaning_count = sum(1 for item in present_keywords if item.get("category") == "meaning")

    locked_categories = set()
    for item in present_keywords:
        category = item.get("category")
        token_count = len(p3._tokenize(item.get("keyword", "")))
        if category == "occasion":
            locked_categories.add(category)
        elif category == "meaning" and (meaning_count >= 2 or token_count >= 2):
            locked_categories.add(category)
        elif category in {"maintenance", "color", "plant_type"} and token_count >= 2:
            locked_categories.add(category)

    return locked_categories


def _build_facet_index(
    flowers: list[dict],
    word_vectorizer: TfidfVectorizer,
    svd: TruncatedSVD | None,
) -> tuple[list[dict], np.ndarray | None, np.ndarray | None]:
    if svd is None:
        return [], None, None

    grouped_flowers: dict[tuple[str, str], list[dict]] = defaultdict(list)
    facet_values: dict[tuple[str, str], str] = {}
    for flower in flowers:
        for category, label in p3._collect_keyword_candidates(flower=flower):
            if category not in FACET_CATEGORY_LIMITS:
                continue

            display_keyword = p3._format_keyword_for_display(category, label)
            normalized_keyword = p3._normalize(display_keyword)
            if not normalized_keyword:
                continue

            key = (category, normalized_keyword)
            grouped_flowers[key].append(flower)
            facet_values.setdefault(key, display_keyword)

    facet_entries = []
    facet_documents = []
    for key, supporting_flowers in grouped_flowers.items():
        category, normalized_keyword = key
        value = facet_values[key]
        facet_document = _build_facet_document(category, value, supporting_flowers)
        facet_word_matrix = word_vectorizer.transform([facet_document])
        if facet_word_matrix.nnz == 0:
            continue

        facet_entries.append(
            {
                "category": category,
                "keyword": p3._format_keyword_for_display(category, value),
                "normalized_keyword": normalized_keyword,
                "support_count": len({flower["name"] for flower in supporting_flowers}),
            }
        )
        facet_documents.append(facet_document)

    if not facet_documents:
        return [], None, None

    facet_word_matrix = word_vectorizer.transform(facet_documents)
    facet_lsa = normalize(svd.transform(facet_word_matrix), norm="l2")
    return facet_entries, facet_word_matrix, facet_lsa


def _semantic_facet_keywords(
    query: str,
    raw_query_word_matrix,
    raw_query_lsa,
    facet_entries: list[dict],
    facet_word_matrix,
    facet_lsa,
    locked_categories: set[str],
    limit: int,
) -> list[dict]:
    if (
        raw_query_word_matrix.nnz == 0
        or facet_word_matrix is None
        or facet_lsa is None
        or not facet_entries
    ):
        return []

    semantic_scores = np.maximum(cosine_similarity(raw_query_lsa, facet_lsa)[0], 0.0)
    lexical_scores = np.maximum(cosine_similarity(raw_query_word_matrix, facet_word_matrix)[0], 0.0)
    combined_scores = FACET_SEMANTIC_WEIGHT * semantic_scores + FACET_LEXICAL_WEIGHT * lexical_scores
    top_score = float(np.max(combined_scores)) if combined_scores.size else 0.0
    if top_score < FACET_MIN_SCORE:
        return []

    normalized_query = p3._normalize(query)
    category_counts: dict[str, int] = defaultdict(int)
    candidate_indices = []
    for index in np.argsort(combined_scores)[::-1]:
        score = float(combined_scores[index])
        if score <= 0:
            break
        if score < FACET_MIN_SCORE or score < top_score * FACET_MIN_RELATIVE_SCORE:
            break

        facet = facet_entries[index]
        if facet["category"] not in FACET_CATEGORY_LIMITS:
            continue
        if facet["category"] in locked_categories:
            continue
        if not _facet_query_is_supported(facet["category"], query):
            continue
        if _keyword_in_query(facet["keyword"], normalized_query):
            continue
        if category_counts[facet["category"]] >= FACET_CATEGORY_LIMITS.get(facet["category"], 1):
            continue
        if (
            facet["category"] in {"meaning", "occasion"}
            and facet["support_count"] <= 1
            and score < top_score * 0.82
        ):
            continue

        candidate_indices.append(index)

    prefer_maintenance = any(
        facet_entries[index]["category"] == "maintenance"
        for index in candidate_indices
    )

    category_counts = defaultdict(int)
    keywords = []
    for index in candidate_indices:
        facet = facet_entries[index]
        if prefer_maintenance and facet["category"] in {"meaning", "occasion"}:
            continue

        category_counts[facet["category"]] += 1
        keywords.append(
            {
                "keyword": facet["keyword"],
                "category": facet["category"],
                "score": round(7.0 + float(combined_scores[index]) * 10 + min(facet["support_count"], 4) / 10, 2),
            }
        )
        if len(keywords) >= limit:
            break

    keywords.sort(
        key=lambda item: (
            FACET_CATEGORY_PRIORITY.get(item["category"], 99),
            -float(item["score"]),
            item["keyword"],
        )
    )
    return keywords[:limit]


def _filter_query_keywords_for_facets(query_keywords: list[dict], facet_keywords: list[dict]) -> list[dict]:
    if not facet_keywords:
        return query_keywords

    facet_keys = {
        (item.get("category"), p3._normalize(item.get("keyword", "")))
        for item in facet_keywords
    }
    facet_categories = {item.get("category") for item in facet_keywords}
    filtered_keywords = []
    for item in query_keywords:
        category = item.get("category")
        keyword = item.get("keyword", "")
        normalized_keyword = p3._normalize(keyword)
        token_count = len(p3._tokenize(keyword))
        if (category, normalized_keyword) in facet_keys:
            continue
        if token_count <= 1 and category in facet_categories:
            continue
        if (
            token_count <= 1
            and category in {"meaning", "occasion", "semantic"}
            and "maintenance" in facet_categories
        ):
            continue
        filtered_keywords.append(
            {
                "keyword": keyword,
                "category": category,
                "score": item.get("score", 0),
            }
        )

    return filtered_keywords


def _build_semantic_query_text(query: str, query_keywords: list[dict], facet_keywords: list[dict]) -> tuple[str, list[dict]]:
    merged_keywords = p3._merge_keyword_lists(
        [_filter_query_keywords_for_facets(query_keywords, facet_keywords), facet_keywords],
        p3.MAX_QUERY_KEYWORDS,
    )
    facet_keys = {
        (item.get("category"), p3._normalize(item.get("keyword", "")))
        for item in facet_keywords
    }

    normalized_query = p3._normalize(query)
    augmented_parts = [query.strip()]
    for item in merged_keywords:
        keyword = item["keyword"].strip()
        normalized_keyword = p3._normalize(keyword)
        if not normalized_keyword:
            continue
        if _keyword_in_query(keyword, normalized_query):
            continue

        repeat_count = 2 if (item.get("category"), normalized_keyword) in facet_keys else 1
        augmented_parts.extend([keyword] * repeat_count)

    return " ".join(part for part in augmented_parts if part), merged_keywords


def _fit_lsa(matrix) -> tuple[TruncatedSVD | None, np.ndarray | None]:
    target_components = max(MIN_SVD_COMPONENTS, matrix.shape[0] // 3)
    n_components = min(MAX_SVD_COMPONENTS, target_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if n_components < 1:
        return None, None

    svd = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        random_state=42,
    )
    return svd, normalize(svd.fit_transform(matrix), norm="l2")


def _empty_response(query: str, keywords: list[dict] | None = None) -> dict:
    response = p3._empty_response(query, keywords)
    response["score_scale"] = "unit"
    return response


def _field_value_map(flower: dict) -> dict[str, list[str]]:
    return {
        "color": list(flower["colors"]),
        "maintenance": list(flower["maintenance"]),
        "plant_type": list(flower["plant_types"]),
        "meaning": _meaning_chunks(flower),
        "occasion": _occasion_chunks(flower),
    }


def _semantic_metadata_bonus(query_keywords: list[dict], flower: dict) -> float:
    field_values = _field_value_map(flower)
    bonus = 0.0

    for item in query_keywords:
        category = item.get("category")
        keyword = item.get("keyword", "")
        normalized_keyword = p3._normalize(keyword)
        keyword_tokens = set(p3._tokenize(keyword))
        if not normalized_keyword or not keyword_tokens or category not in field_values:
            continue

        best_match = 0.0
        for value in field_values[category]:
            normalized_value = p3._normalize(value)
            value_tokens = set(p3._tokenize(value))
            if not normalized_value or not value_tokens:
                continue

            if normalized_keyword == normalized_value:
                best_match = max(best_match, 1.0)
                continue

            overlap = len(keyword_tokens & value_tokens)
            if overlap == min(len(keyword_tokens), len(value_tokens)):
                best_match = max(best_match, 0.72)

        if best_match <= 0:
            continue

        category_weight = 0.45 if category in {"meaning", "occasion"} else 0.18
        bonus += category_weight * best_match

    return bonus


@lru_cache(maxsize=1)
def _load_model() -> tuple:
    flowers = p3._build_corpus_docs()
    for flower in flowers:
        passages, document = _build_document(flower)
        flower["passages"] = passages
        flower["document"] = document

    documents = [flower["document"] for flower in flowers]
    word_vectorizer = TfidfVectorizer(
        preprocessor=p3._normalize,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        dtype=np.float32,
    )
    char_vectorizer = TfidfVectorizer(
        preprocessor=p3._normalize,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
        dtype=np.float32,
    )

    word_matrix = word_vectorizer.fit_transform(documents)
    char_matrix = char_vectorizer.fit_transform(documents)
    svd, lsa_matrix = _fit_lsa(word_matrix)
    facet_entries, facet_word_matrix, facet_lsa = _build_facet_index(flowers, word_vectorizer, svd)
    component_labels = p3._component_labels_from_svd(svd, word_vectorizer, flowers)

    return (
        flowers,
        word_vectorizer,
        char_vectorizer,
        word_matrix,
        int(word_matrix.shape[1] + char_matrix.shape[1]),
        svd,
        lsa_matrix,
        component_labels,
        facet_entries,
        facet_word_matrix,
        facet_lsa,
    )


def model_info() -> dict:
    flowers, word_vectorizer, char_vectorizer, _, feature_count, svd, _, _, facet_entries, _, _ = _load_model()
    return {
        "retrieval_mode": "semantic_csv_svd",
        "document_count": len(flowers),
        "feature_count": feature_count,
        "word_feature_count": len(word_vectorizer.get_feature_names_out()),
        "char_feature_count": len(char_vectorizer.get_feature_names_out()),
        "semantic_facet_count": len(facet_entries),
        "svd_components": 0 if svd is None else int(svd.n_components),
        "svd_explained_variance_sum": 0.0 if svd is None else float(svd.explained_variance_ratio_.sum()),
    }


def semantic_diagnostics(queries: tuple[str, ...] = SEMANTIC_BENCHMARK_QUERIES) -> dict:
    (
        flowers,
        word_vectorizer,
        _,
        word_matrix,
        _,
        svd,
        lsa_matrix,
        _,
        facet_entries,
        facet_word_matrix,
        facet_lsa,
    ) = _load_model()
    if svd is None or lsa_matrix is None:
        return {
            "query_count": 0,
            "svd_differs_from_lexical_top3": 0,
            "svd_differs_from_lexical_top1": 0,
            "explained_variance_sum": 0.0,
            "examples": [],
        }

    different_top3 = 0
    different_top1 = 0
    examples = []

    for query in queries:
        raw_query_word_matrix = word_vectorizer.transform([query])
        if raw_query_word_matrix.nnz == 0:
            continue
        raw_query_lsa = normalize(svd.transform(raw_query_word_matrix), norm="l2")
        base_query_keywords = p3._build_query_breakdown_keywords(
            query,
            raw_query_word_matrix,
            word_vectorizer,
            flowers,
            p3.MAX_QUERY_KEYWORDS,
        )
        locked_categories = _locked_facet_categories(query, base_query_keywords)
        facet_keywords = _semantic_facet_keywords(
            query,
            raw_query_word_matrix,
            raw_query_lsa,
            facet_entries,
            facet_word_matrix,
            facet_lsa,
            locked_categories,
            MAX_QUERY_EXPANSIONS,
        )
        semantic_query_text, _ = _build_semantic_query_text(query, base_query_keywords, facet_keywords)
        query_word_matrix = word_vectorizer.transform([semantic_query_text])

        lexical_scores = np.maximum(cosine_similarity(query_word_matrix, word_matrix)[0], 0.0)
        semantic_scores = np.maximum(cosine_similarity(normalize(svd.transform(query_word_matrix), norm="l2"), lsa_matrix)[0], 0.0)

        lexical_top = [flowers[index]["name"] for index in np.argsort(lexical_scores)[::-1][:3] if lexical_scores[index] > 0]
        semantic_top = [flowers[index]["name"] for index in np.argsort(semantic_scores)[::-1][:3] if semantic_scores[index] > 0]

        if lexical_top[:1] != semantic_top[:1]:
            different_top1 += 1
        if lexical_top != semantic_top:
            different_top3 += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "query": query,
                        "lexical_top3": lexical_top,
                        "svd_top3": semantic_top,
                    }
                )

    return {
        "query_count": len(queries),
        "svd_differs_from_lexical_top3": different_top3,
        "svd_differs_from_lexical_top1": different_top1,
        "explained_variance_sum": float(svd.explained_variance_ratio_.sum()),
        "examples": examples,
    }


def recommend_flowers(query: str, limit: int = 5) -> dict:
    if not query or not query.strip():
        return _empty_response(query)

    (
        flowers,
        word_vectorizer,
        char_vectorizer,
        word_matrix,
        _,
        svd,
        lsa_matrix,
        component_labels,
        facet_entries,
        facet_word_matrix,
        facet_lsa,
    ) = _load_model()
    raw_query_word_matrix = word_vectorizer.transform([query])

    if raw_query_word_matrix.nnz == 0 or svd is None or lsa_matrix is None:
        return _empty_response(query)

    raw_query_lsa = normalize(svd.transform(raw_query_word_matrix), norm="l2")
    base_query_keywords = p3._build_query_breakdown_keywords(
        query,
        raw_query_word_matrix,
        word_vectorizer,
        flowers,
        p3.MAX_QUERY_KEYWORDS,
    )
    locked_categories = _locked_facet_categories(query, base_query_keywords)
    facet_keywords = _semantic_facet_keywords(
        query,
        raw_query_word_matrix,
        raw_query_lsa,
        facet_entries,
        facet_word_matrix,
        facet_lsa,
        locked_categories,
        MAX_QUERY_EXPANSIONS,
    )
    semantic_query_text, query_keywords = _build_semantic_query_text(query, base_query_keywords, facet_keywords)
    query_word_matrix = word_vectorizer.transform([semantic_query_text])

    if query_word_matrix.nnz == 0:
        return _empty_response(query, query_keywords)

    query_lsa = normalize(svd.transform(query_word_matrix), norm="l2")
    semantic_similarities = np.maximum(cosine_similarity(query_lsa, lsa_matrix)[0], 0.0)
    direct_word_similarities = np.maximum(cosine_similarity(query_word_matrix, word_matrix)[0], 0.0)
    display_similarities = (
        SEMANTIC_SIMILARITY_WEIGHT * semantic_similarities
        + LEXICAL_STABILITY_WEIGHT * direct_word_similarities
    )
    similarities = np.asarray(
        [
            float(display_similarities[index]) + _semantic_metadata_bonus(query_keywords, flowers[index])
            for index in range(len(flowers))
        ],
        dtype=np.float32,
    )
    ranked_indices = np.argsort(similarities)[::-1]
    positive_scores = [float(similarities[index]) for index in ranked_indices if float(similarities[index]) > 0]
    if not positive_scores:
        return _empty_response(query, query_keywords)

    query_axes = p3._select_query_latent_axes(
        query_lsa[0],
        component_labels,
        query_keywords,
        svd,
        word_vectorizer,
    )
    query_radar_chart = None
    query_axis_indices = np.asarray([], dtype=int)
    query_axis_labels: list[str] = []
    if query_axes is not None:
        query_axis_indices = query_axes["axis_indices"]
        query_axis_labels = p3._relabel_query_axes(
            query_axis_indices,
            query_axes["axis_labels"],
            query_keywords,
            svd,
            word_vectorizer,
        )
        query_axis_display_values = p3._query_axis_display_values(
            query_lsa[0],
            query_axis_indices,
            query_axis_labels,
            query_keywords,
        )
        query_radar_chart = build_latent_radar_chart(
            query_lsa[0],
            component_labels,
            profile_kind="query",
            axis_indices=query_axis_indices,
            axis_labels=query_axis_labels,
            axis_values=query_axis_display_values,
        )

    suggestions = []
    for index in ranked_indices:
        similarity = float(similarities[index])
        if len(suggestions) >= limit or similarity <= 0:
            break

        suggestion = p3._build_suggestion(
            flowers[index],
            query,
            similarity,
            query_lsa[0],
            lsa_matrix[index],
            query_word_matrix,
            word_matrix,
            index,
            word_vectorizer,
            char_vectorizer,
            svd,
            component_labels,
            query_axis_indices,
            query_axis_labels,
        )
        suggestion["score"] = round(float(similarity / (1.0 + similarity)), 4)
        suggestions.append(suggestion)

    return {
        "query": query,
        "keywords_used": query_keywords,
        "score_scale": "unit",
        "query_latent_radar_chart": None if query_radar_chart is None else query_radar_chart["image_data_url"],
        "query_latent_radar_axes": [] if query_radar_chart is None else query_radar_chart["axis_labels"],
        "suggestions": suggestions,
    }


def visualizer_flowers(limit: int = 48) -> dict:
    flowers, _, _, _, _, _, lsa_matrix, component_labels, _, _, _ = _load_model()
    if lsa_matrix is None or len(flowers) == 0:
        return {"flowers": []}

    ranked_indices = sorted(
        range(len(flowers)),
        key=lambda index: (
            p3._metadata_richness(flowers[index])[0],
            p3._metadata_richness(flowers[index])[1],
            flowers[index]["name"].lower(),
        ),
        reverse=True,
    )
    chosen_indices = ranked_indices[: max(1, min(limit, len(ranked_indices)))]
    chosen_vectors = np.asarray([lsa_matrix[index] for index in chosen_indices], dtype=np.float32)
    normalized_positions = p3._normalize_visualizer_positions(chosen_vectors)

    dataset = []
    for output_index, flower_index in enumerate(chosen_indices):
        flower = flowers[flower_index]
        axis_info = select_latent_axes(lsa_matrix[flower_index], component_labels)
        x, y, z = normalized_positions[output_index]
        dataset.append(
            {
                "id": p3._normalize(flower["name"]).replace(" ", "-"),
                "name": flower["name"],
                "scientific_name": flower["scientific_name"],
                "colors": flower["colors"],
                "plant_types": flower["plant_types"],
                "maintenance": p3._format_maintenance_values(flower["maintenance"]),
                "meanings": flower["meanings"],
                "occasions": flower["occasions"],
                "primary_color": flower["colors"][0] if flower["colors"] else "neutral",
                "primary_meaning": flower["meanings"][0] if flower["meanings"] else "general",
                "primary_occasion": flower["occasions"][0] if flower["occasions"] else "everyday",
                "latent_axes": [] if axis_info is None else axis_info["axis_labels"],
                "latent_position": {
                    "x": round(float(x), 4),
                    "y": round(float(y), 4),
                    "z": round(float(z), 4),
                },
                "image_url": flower.get("image_url"),
                "summary": flower["passages"][:3],
            }
        )

    return {"flowers": dataset}


def get_flower_vectors(scientific_names: list[str]) -> dict:
    flowers, _, _, _, _, _, lsa_matrix, _, _, _, _ = _load_model()
    sci_to_index = {
        p3._normalize(f["scientific_name"]): i
        for i, f in enumerate(flowers)
    }

    result = []
    for name in scientific_names:
        key = p3._normalize(name)
        if key in sci_to_index:
            index = sci_to_index[key]
            result.append(
                {
                    "index": index,
                    "name": flowers[index]["name"],
                    "scientific_name": flowers[index]["scientific_name"],
                    "meanings": flowers[index]["meanings"],
                    "colors": flowers[index]["colors"],
                    "maintenance": p3._format_maintenance_values(flowers[index]["maintenance"]),
                    "plant_types": flowers[index]["plant_types"],
                    "occasions": flowers[index]["occasions"],
                    "lsa_vector": lsa_matrix[index],
                }
            )

    return {
        "flowers": result,
        "total_flower_count": len(flowers),
    }
