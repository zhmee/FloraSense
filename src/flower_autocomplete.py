"""
Autocomplete for FloraSense.

This module keeps autocomplete separate from the main recommender so
`flower_recommender_prototype3.py` does not keep growing into a monster.

High-level flow:
1. Build a list of short, query-shaped phrases from flower metadata.
2. Convert those phrases into TF-IDF features.
3. Optionally compress the features with SVD / LSA.
4. Rank phrases for a user query with semantic similarity plus prefix matching.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from flower_recommender_prototype3 import (
    MAX_SVD_COMPONENTS,
    _build_corpus_docs,
    _combined_features,
    _dedupe_preserve_order,
    _fit_lsa,
    _normalize,
    _split_meaning_chunks,
    _tokenize,
)

# keep the autocomplete responses short so the dropdown stays readable
MAX_AUTOCOMPLETE_SUGGESTIONS = 6

# these are the prompt shapes we want suggestions to look like
QUERY_PREFIX_TEMPLATES = ("flowers for ", "a flower for ")

# these specific tokens usually produce awkward or low-information generated prompts,
# so we skip them when building autocomplete phrases from meanings/occasions
LOW_SIGNAL_LEADING_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "flower",
    "flowers",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "this",
    "that",
    "these",
    "those",
    "to",
    "with",
}


def _format_autocomplete_phrase(phrase: str) -> str:
    """
    Clean up a phrase before storing or returning it.

    This gives us lowercase text with normalized whitespace so the same idea
    is not stored in multiple slightly different forms.
    """
    return " ".join((phrase or "").strip().lower().split())


def _autocomplete_prefix_score(query: str, phrase: str) -> float:
    """
    Score how much a phrase looks like a direct continuation of what the user typed.

    Semantic similarity is the main ranking signal, but autocomplete also needs to
    feel responsive while the user is still in the middle of typing a word.
    """
    # normalize both inputs so comparison logic is case-insensitive and consistent
    normalized_query = _normalize(query)
    normalized_phrase = _normalize(phrase)
    if not normalized_query or not normalized_phrase:
        return 0.0

    # exact prefix matches get the strongest boost
    if normalized_phrase.startswith(normalized_query):
        return 1.0

    # xcontainment still matters, but not as much as a true prefix continuation
    if normalized_query in normalized_phrase:
        return 0.45

    # token-based checks help with partially typed words like "low ma" (hehe yo mama)
    query_tokens = normalized_query.split()
    phrase_tokens = normalized_phrase.split()
    if not query_tokens or not phrase_tokens:
        return 0.0

    # for a single-token query, reward any phrase token that starts the same way
    if len(query_tokens) == 1:
        return 0.8 if any(token.startswith(query_tokens[0]) for token in phrase_tokens) else 0.0

    # for multi-token queries, require the earlier tokens to be present and allow
    # the final token to be only partially typed
    fixed_tokens = query_tokens[:-1]
    partial_token = query_tokens[-1]
    if all(token in phrase_tokens for token in fixed_tokens) and any(token.startswith(partial_token) for token in phrase_tokens):
        return 0.8
    return 0.0


def _strict_prefix_matches(query: str, phrases: list[str]) -> list[str]:
    """
    Return phrases that literally start with the exact normalized query.

    If the user is already typing a known phrase, we can answer immediately
    without asking the semantic model to guess.
    """
    normalized_query = _format_autocomplete_phrase(query)
    if not normalized_query:
        return []
    return [phrase for phrase in phrases if phrase.startswith(normalized_query)]


def _prefix_template_matches(query: str, phrases: list[str]) -> list[str]:
    """
    Preserve the exact prompt template the user has started typing.

    Example:
    - stored phrase: "flowers for courage"
    - typed query: "a flower for c"
    - returned suggestion: "a flower for courage"

    This makes autocomplete feel consistent even when we can borrow the tail
    from a phrase stored under a different template.
    """
    normalized_query = _format_autocomplete_phrase(query)
    for template in QUERY_PREFIX_TEMPLATES:
        if not normalized_query.startswith(template):
            continue

        # everything after the template is the part the user is actively typing
        partial_tail = normalized_query[len(template):].strip()
        if not partial_tail:
            return [phrase for phrase in phrases if phrase.startswith(template)]

        rewritten = []
        for phrase in phrases:
            for phrase_template in QUERY_PREFIX_TEMPLATES:
                if not phrase.startswith(phrase_template):
                    continue
                # reuse the stored phrase tail, but keep the exact prompt prefix the user typed
                tail = phrase[len(phrase_template):]
                if tail.startswith(partial_tail):
                    rewritten.append(f"{template}{tail}")
        # deduping matters here because different stored templates may rewrite
        # to the same final suggestion
        return _dedupe_preserve_order(rewritten)

    return []


def _short_query_phrases(texts: list[str], prefix: str) -> list[str]:
    """
    Turn raw metadata text into short autocomplete prompts.

    Example:
    - input metadata: "love; devotion"
    - output prompts: "flowers for love", "flowers for devotion"
    """
    phrases = []
    for text in texts:
        # the recommender stores some fields as larger meaning/occasion strings
        # break those into smaller pieces before turning them into suggestions (CHUNKING I THINK)
        for chunk in _split_meaning_chunks(text):
            normalized_chunk = _format_autocomplete_phrase(chunk)
            candidate_chunks = [normalized_chunk]
            # split compound metadata so one record can produce several focused prompts
            candidate_chunks.extend(
                _format_autocomplete_phrase(part)
                for part in re.split(r"\b(?:and|or)\b|,|/|:", normalized_chunk)
            )

            for candidate in _dedupe_preserve_order(candidate_chunks):
                tokens = _tokenize(candidate)
                # keep prompts compact so the dropdown remains readable
                if not 1 <= len(tokens) <= 6:
                    continue
                # skip phrases that begin with generic filler words
                if tokens[0] in LOW_SIGNAL_LEADING_TOKENS:
                    continue
                # autocomplete suggestions should read like a search prompt, not raw metadata
                phrases.append(_format_autocomplete_phrase(f"{prefix} {candidate}"))
    return _dedupe_preserve_order(phrases)


def _should_keep_generated_phrase(phrase: str, frequency: int) -> bool:
    """
    Decide whether a generated prompt is useful enough to keep.

    The phrase builder creates a lot of candidates. This filter removes prompts
    that are too long, too short, or too rare to look trustworthy in the UI.
    """
    tokens = _tokenize(phrase)
    if not 1 <= len(tokens) <= 8:
        return False

    if phrase.startswith("flowers for "):
        # "flowers for courage" is fine, but a one-word tail like
        # "flowers for gay" is usually noise unless it appears more than once
        tail_tokens = tokens[2:]
        return len(tail_tokens) >= 2 or frequency >= 2

    if phrase.startswith("a flower for "):
        # same filtering rule for the alternate prompt template
        tail_tokens = tokens[3:]
        return len(tail_tokens) >= 2 or frequency >= 2

    return True


def _build_autocomplete_phrases(flowers: list[dict]) -> list[str]:
    """
    Build the full bank of phrases that autocomplete is allowed to return.

    We intentionally avoid suggesting flower names directly. The autocomplete box
    is meant to help the user describe what they want, not jump straight to a
    specific flower unless the recommender later decides that flower is a match (otherwise
    they wouldn't need to press enter it would just automatically give a flower).
    """
    direct_phrases: list[str] = []
    generated_phrases: list[str] = []
    for flower in flowers:
        # keep direct trait queries that users are likely to type verbatim
        for color in flower["colors"]:
            direct_phrases.append(_format_autocomplete_phrase(f"{color} flowers"))
            for maintenance in flower["maintenance"]:
                direct_phrases.append(_format_autocomplete_phrase(f"{maintenance} maintenance {color} flowers"))

        for maintenance in flower["maintenance"]:
            direct_phrases.append(_format_autocomplete_phrase(f"{maintenance} maintenance flowers"))

        # turn meanings and occasions into short prompt-shaped suggestions
        generated_phrases.extend(_short_query_phrases(flower["meanings"], "flowers for"))
        generated_phrases.extend(_short_query_phrases(flower["meanings"], "a flower for"))
        generated_phrases.extend(_short_query_phrases(flower["occasions"], "flowers for"))

    # generated prompts can repeat across multiple flower records, which is useful
    # because repeated phrases are usually more trustworthy than one-off phrases
    generated_counts = Counter(generated_phrases)
    cleaned_generated = [
        phrase
        for phrase in _dedupe_preserve_order(generated_phrases)
        if _should_keep_generated_phrase(phrase, generated_counts[phrase])
    ]
    # Keep the original insertion order while removing duplicates.
    return _dedupe_preserve_order(direct_phrases + cleaned_generated)


@lru_cache(maxsize=1)
def _load_autocomplete_model() -> tuple:
    """
    Build and cache the autocomplete search index.

    This is cached because the phrase list and its vector representations are
    expensive to rebuild on every request.
    """
    # load the flower records from the main recommender pipeline
    flowers = _build_corpus_docs()

    # convert flower metadata into the phrases users will see in autocomplete
    phrases = _build_autocomplete_phrases(flowers)

    # word n-grams capture semantic prompt overlap
    word_vectorizer = TfidfVectorizer(
        preprocessor=_normalize,
        stop_words="english",
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        dtype=np.float32,
    )
    # character n-grams preserve partial-word matching for incomplete typing
    char_vectorizer = TfidfVectorizer(
        preprocessor=_normalize,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
        dtype=np.float32,
    )

    # build both feature spaces, then combine them into one matrix so ranking can
    # benefit from semantic phrase overlap and partial-word overlap at the same time
    word_matrix = word_vectorizer.fit_transform(phrases)
    char_matrix = char_vectorizer.fit_transform(phrases)
    combined_matrix = hstack([word_matrix, char_matrix], format="csr")

    # compress the feature matrix when possible so similarity search is cheaper
    # and can capture broader semantic structure
    svd, lsa_matrix = _fit_lsa(combined_matrix, min(MAX_SVD_COMPONENTS, 64))
    return phrases, word_vectorizer, char_vectorizer, svd, lsa_matrix


def autocomplete_queries(query: str, limit: int = MAX_AUTOCOMPLETE_SUGGESTIONS) -> dict:
    """
    Return the best autocomplete suggestions for a user query.

    Ranking uses two signals:
    1. semantic similarity from the vector model
    2. a smaller bonus for phrases that directly continue the typed text
    """
    # empty input should not produce suggestions
    if not query or not query.strip():
        return {"query": query, "suggestions": []}

    # load the phrase bank and the fitted feature models
    phrases, word_vectorizer, char_vectorizer, svd, lsa_matrix = _load_autocomplete_model()
    template_matches = _prefix_template_matches(query, phrases)
    if template_matches:
        # if we can preserve the exact typed prompt shape, do that before semantic ranking
        return {
            "query": query,
            "suggestions": template_matches[:limit],
        }

    strict_prefix_matches = _strict_prefix_matches(query, phrases)
    if strict_prefix_matches:
        # literal continuations should win when the user is already on a known phrase
        return {
            "query": query,
            "suggestions": strict_prefix_matches[:limit],
        }

    # convert the raw query into the same combined feature space as the phrase bank
    query_matrix = _combined_features([query], word_vectorizer, char_vectorizer)
    if query_matrix.nnz == 0:
        # fall back to prefix-only matching if every query token is out of vocabulary
        prefix_matches = [phrase for phrase in phrases if _autocomplete_prefix_score(query, phrase) > 0]
        return {"query": query, "suggestions": prefix_matches[:limit]}

    if svd is None or lsa_matrix is None:
        # if SVD is unavailable, compare directly in the original sparse TF-IDF space
        phrase_matrix = _combined_features(phrases, word_vectorizer, char_vectorizer)
        similarities = cosine_similarity(query_matrix, phrase_matrix)[0]
    else:
        # otherwise compare in the lower-dim LSA space
        query_lsa = normalize(svd.transform(query_matrix), norm="l2")
        similarities = cosine_similarity(query_lsa, lsa_matrix)[0]

    scored_phrases = []
    for index, phrase in enumerate(phrases):
        semantic_score = max(0.0, float(similarities[index]))
        prefix_score = _autocomplete_prefix_score(query, phrase)
        # semantic similarity drives ranking, with a smaller boost for typed-prefix continuity
        total_score = semantic_score + 0.35 * prefix_score
        if total_score > 0:
            scored_phrases.append((total_score, prefix_score, phrase))

    # prefer higher scores first, then stronger prefix matches, then shorter phrases
    scored_phrases.sort(key=lambda item: (-item[0], -item[1], len(item[2]), item[2].lower()))
    return {
        "query": query,
        # normalize one last time before returning to the frontend
        "suggestions": [_format_autocomplete_phrase(phrase) for _, _, phrase in scored_phrases[:limit]],
    }
