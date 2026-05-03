"""
FloraSense retrieval: inverted index, TF-IDF, and Rocchio feedback.
Dataset: merged.csv for metadata and merged_preprocessed_meanings.csv for
         preprocessed meaning/occasion card text.

IR concepts applied
-------------------
- Basic text processing      : normalization, tokenization, stemming
                               (PorterStemmer), stopword removal,
                               de-duplication via sets
- Inverted index             : postings list per term for exact phrase
                               matching and keyword categorization
                               (longest-match-first traversal)
- TF-IDF weighting           : sublinear TF scaling via TfidfVectorizer;
                               field importance via token repetition
- Term-document matrix       : sparse (n_flowers x n_terms) built by
                               TfidfVectorizer.fit_transform()
- Cosine similarity          : l2-normalized dot product for ranking
- Rocchio's method           : pseudo-relevance feedback shifts query
                               vector toward centroid of top-k results
- Minimum Edit Distance      : Wagner-Fischer DP for fuzzy color/maintenance
                               typo handling
"""

import csv
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from utils import resolve_flower_image_url, split_meaning_cell

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).resolve().parent / "data" / "merged.csv"
PREPROCESSED_MEANINGS_FILE = Path(__file__).resolve().parent / "data" / "merged_preprocessed_meanings.csv"

# Token repetition per field: raises raw TF so TF-IDF weights that field more.
FIELD_REPEAT = {
    "name":            3,
    "meaning":         3,
    "color":           2,
    "maintenance":     2,
    "occasions":       2,
    "plant_type":      1,
    "scientific_name": 1,
}

# Base weights for the inverted index scoring (query breakdown display only)
BASE_WEIGHTS = {
    "common_name":    9.0,
    "scientific_name": 7.5,
    "color":          4.5,
    "maintenance":    4.0,
    "plant_type":     3.5,
    "meaning_phrase": 5.5,
    "meaning_token":  2.2,
    "occasion_token": 2.0,
}

CATEGORY_LABELS = {
    "common_name":     "name",
    "scientific_name": "scientific name",
    "color":           "color",
    "maintenance":     "maintenance",
    "plant_type":      "plant type",
    "meaning_phrase":  "meaning",
    "meaning_token":   "meaning",
    "occasion_token":  "occasion",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "best", "by", "for", "from", "i",
    "in", "is", "it", "like", "me", "my", "myself", "of", "or", "our",
    "ours", "show", "something", "that", "the", "to", "want", "we", "with",
    # verb connectors -- structural query words, not content
    "mean", "means", "meaning", "meanings", "called", "known", "named",
    # filler in a flower app -- everything is a flower
    "flower", "flowers", "bloom", "blooms", "plant", "plants",
}

# Rocchio pseudo-relevance feedback parameters
ROCCHIO_ALPHA = 1.0   # original query weight
ROCCHIO_BETA  = 0.75  # relevant documents weight
ROCCHIO_TOP_K = 3     # how many top results to treat as pseudo-relevant

# MED: max edit distance allowed for fuzzy color/maintenance matching
MED_THRESHOLD = 2

MAX_MATCHED_CHIPS  = 8
MAX_QUERY_KEYWORDS = 10

# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_stemmer = PorterStemmer()


def _normalize(text: str) -> str:
    """
    Lowercase and strip punctuation.
    """
    return re.sub(r"[^a-z0-9\s]+", " ", text.lower()).strip()


def _tokenize(text: str) -> list:
    """
    Normalize and split into tokens (no stemming -- for inverted index).
    """
    return re.findall(r"[a-z0-9]+", _normalize(text))


def _tokenize_and_stem(text: str) -> list:
    """
    Normalize, tokenize, remove stopwords, then stem.

    Used as the TfidfVectorizer analyzer so corpus documents and queries
    share the same tokenization and vocabulary.
    """
    tokens = re.findall(r"[a-z0-9]+", re.sub(r"[^a-z0-9\s]+", " ", text.lower()))
    return [
        _stemmer.stem(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]


def _split_csv_cell(value: str) -> list:
    return [p.strip() for p in value.split(",") if p.strip()]


def _split_meanings(value: str) -> list:
    """
    Split meaning entries separated by semicolons.
    """
    return split_meaning_cell(value)


def _dedupe_sorted(values) -> list[str]:
    return sorted({value.strip() for value in values if value and value.strip()})


def _preprocessed_lookup_key(name: str, color: str = "") -> str:
    return f"{_normalize(name)}::{_normalize(color)}"


@lru_cache(maxsize=1)
def _load_preprocessed_meaning_texts() -> dict[str, dict[str, list[str]]]:
    summaries: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"meanings": [], "occasions": []}
    )
    if not PREPROCESSED_MEANINGS_FILE.exists():
        return {}

    with PREPROCESSED_MEANINGS_FILE.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("name") or "").strip()
            color = (row.get("color") or "").strip()
            if not name:
                continue
            meaning = (row.get("ir_summary") or "").strip()
            occasion = (row.get("occasions_summary") or "").strip()
            for key in (_preprocessed_lookup_key(name, color), _preprocessed_lookup_key(name)):
                if meaning:
                    summaries[key]["meanings"].append(meaning)
                if occasion:
                    summaries[key]["occasions"].append(occasion)

    return {
        key: {
            "meanings": _dedupe_sorted(value["meanings"]),
            "occasions": _dedupe_sorted(value["occasions"]),
        }
        for key, value in summaries.items()
    }


def _preprocessed_text_for_row(row: dict) -> dict[str, list[str]]:
    lookup = _load_preprocessed_meaning_texts()
    name = (row.get("name") or "").strip()
    color = (row.get("color") or "").strip()
    return (
        lookup.get(_preprocessed_lookup_key(name, color))
        or lookup.get(_preprocessed_lookup_key(name))
        or {"meanings": [], "occasions": []}
    )


def _plant_type_aliases(value: str) -> set:
    """
    Singular/plural aliases so 'annual' matches 'Annuals'.
    """
    lowered = value.lower().strip()
    aliases = {lowered}
    if lowered.endswith("ies") and len(lowered) > 3:
        aliases.add(lowered[:-3] + "y")
    elif lowered.endswith("s") and len(lowered) > 3:
        aliases.add(lowered[:-1])
    return aliases

# ---------------------------------------------------------------------------
# TF-IDF Ranking
# ---------------------------------------------------------------------------

def _rank_tfidf(query_tfidf, td_matrix):
    # Ensure dense 2D
    if hasattr(query_tfidf, "toarray"):
        query_vec = query_tfidf.toarray()
    else:
        query_vec = np.atleast_2d(query_tfidf)

    docs = td_matrix.toarray()

    query_norm = normalize(query_vec)
    docs_norm = normalize(docs)

    return docs_norm.dot(query_norm.T).squeeze()


# ----------------------------------------------------------------------------
# Minimum Edit Distance (Wagner-Fischer)
# ----------------------------------------------------------------------------

def _med(s: str, t: str) -> int:
    """
    Wagner-Fischer dynamic programming for minimum edit distance.

    dp[i][j] = edit distance between s[:i] and t[:j].
    Operations: insertion, deletion, substitution (each cost 1).

    Used to fuzzy-match user-typed tokens against known corpus values
    so typos like 'purpl' -> 'purple' or 'maintanence' -> 'maintenance'
    still categorize correctly in the query breakdown.

    Only for COLOR and MAINTENANCE for safety against false positives.
    (Edit distance has no semantic awareness, so "love" -> "live" is 
     just as likely as "lvoe" -> "love", unfortunately.)
    """
    m, n = len(s), len(t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[j] = prev[j - 1]       # match: no cost
            else:
                dp[j] = 1 + min(
                    prev[j],              # deletion
                    dp[j - 1],            # insertion
                    prev[j - 1],          # substitution
                )
    return dp[n]


def _fuzzy_match(token: str, known_values: list):
    """
    Return the closest known value within MED_THRESHOLD, or None.
    Only used for short structured fields (colors, maintenance) where
    typos are likely but semantic drift is not.
    """
    best, best_d = None, MED_THRESHOLD + 1
    for val in known_values:
        d = _med(token, _normalize(val))
        if d < best_d:
            best_d, best = d, val
    return best if best_d <= MED_THRESHOLD else None


# ---------------------------------------------------------------------------
# Document builder  (for TF-IDF term-document matrix)
# ---------------------------------------------------------------------------

def _build_document(flower: dict) -> str:
    """
    Concatenate all flower fields into one weighted text string.
    Token repetition encodes field priority directly into raw TF so
    TF-IDF naturally amplifies more important fields without custom math.
    This string is what becomes one row in the term-document matrix.

    Uses the FULL meanings and occasions (including long prose) for
    maximum vocabulary coverage for TF-IDF ranking and for the API
    response (same strings the UI shows with click-to-expand).
    """
    parts = []
    parts += [_normalize(flower["name"])]            * FIELD_REPEAT["name"]
    parts += [_normalize(flower["scientific_name"])] * FIELD_REPEAT["scientific_name"]
    for c in flower["colors"]:
        parts += [_normalize(c)] * FIELD_REPEAT["color"]
    for m in flower["maintenance"]:
        parts += [_normalize(m)] * FIELD_REPEAT["maintenance"]
    for meaning in flower["meanings"]:
        parts += [_normalize(meaning)] * FIELD_REPEAT["meaning"]
    for occ in flower["occasions"]:
        parts += [_normalize(occ)] * FIELD_REPEAT["occasions"]
    for pt in flower["plant_types"]:
        parts += [_normalize(pt)] * FIELD_REPEAT["plant_type"]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Model loader  (lru_cache -- built once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model() -> tuple:
    """
    Load merged CSV, build flower records, inverted index for keyword display,
    TF-IDF vectorizer and term-document matrix, and auxiliary maps for fuzzy
    matching and chip labels. Cached once per process via lru_cache.
    """

    # ── 1. Load CSV and group multi-row flowers ──────────────────────────────
    # merged.csv column names:
    #   'name'             (common name)
    #   'Special Occasions' (occasion prose)
    grouped: dict = {}

    with DATA_FILE.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            common = row.get("name", "").strip()
            sci    = row.get("scientific_name", "").strip()
            key    = _normalize(common or sci)
            if not key:
                continue
            if key not in grouped:
                grouped[key] = {
                    "key":              key,
                    "name":             common or sci or "Unknown",
                    "scientific_name":  sci or "Unknown",
                    "colors":           set(),
                    "plant_types":      set(),
                    "maintenance":      set(),
                    "meanings":         set(),
                    "occasions":        set(),
                }
            e = grouped[key]
            if row.get("color", "").strip():
                e["colors"].add(row["color"].strip())
            if row.get("maintenance", "").strip():
                e["maintenance"].add(row["maintenance"].strip())
            for pt in _split_csv_cell(row.get("planttype", "")):
                e["plant_types"].add(pt)

            preprocessed_text = _preprocessed_text_for_row(row)
            e["meanings"].update(preprocessed_text["meanings"])
            e["occasions"].update(preprocessed_text["occasions"])

    # Convert sets -> sorted lists for deterministic output
    flowers = [
        {
            "key":               e["key"],
            "name":              e["name"],
            "scientific_name":   e["scientific_name"],
            "colors":            sorted(e["colors"]),
            "plant_types":       sorted(e["plant_types"]),
            "maintenance":       sorted(e["maintenance"]),
            "meanings":          sorted(e["meanings"]),
            "occasions":         sorted(e["occasions"]),
        }
        for e in grouped.values()
    ]

    # ── 2. Known field values for MED fuzzy matching ─────────────────────────
    known_colors = list({c for f in flowers for c in f["colors"]})
    known_maint  = list({m for f in flowers for m in f["maintenance"]})

    # ── 3. Stem -> display word map ──────────────────────────────────────────
    # First occurrence of each stem wins so chips show a readable surface form
    # (e.g. 'everlasting') rather than the stem alone ('everlast').
    stem_to_word: dict = {}
    for flower in flowers:
        all_text = " ".join([
            flower["name"], flower["scientific_name"],
            *flower["colors"], *flower["maintenance"],
            *flower["plant_types"], *flower["meanings"], *flower["occasions"],
        ])
        for tok in re.findall(r"[a-z]+", _normalize(all_text)):
            if len(tok) > 2 and tok not in STOPWORDS:
                stem_to_word.setdefault(_stemmer.stem(tok), tok)

    # ── 4. Inverted index (for keyword display only) ─────────────────────────
    # Separate from TF-IDF. Stores postings: term -> [{flower_key, category}]
    # De-duplicated via seen set so no flower is double-counted per term.
    keyword_index: dict = defaultdict(list)
    seen: set = set()

    for flower in flowers:
        entries = []
        entries.append((_normalize(flower["name"]), "common_name"))
        entries.append((_normalize(flower["scientific_name"]), "scientific_name"))
        for c in flower["colors"]:
            entries.append((_normalize(c), "color"))
        for m in flower["maintenance"]:
            entries.append((_normalize(m), "maintenance"))
        for pt in flower["plant_types"]:
            for alias in _plant_type_aliases(pt):
                entries.append((alias, "plant_type"))
        for meaning in flower["meanings"]:
            entries.append((_normalize(meaning), "meaning_phrase"))
            for tok in _tokenize(meaning):
                if len(tok) > 2 and tok not in STOPWORDS:
                    entries.append((tok, "meaning_token"))
        for occ in flower["occasions"]:
            for tok in _tokenize(occ):
                if len(tok) > 2 and tok not in STOPWORDS:
                    entries.append((tok, "occasion_token"))

        for term, category in entries:
            if not term or term in STOPWORDS:
                continue
            idx_key = (term, category, flower["key"])
            if idx_key in seen:
                continue
            seen.add(idx_key)
            keyword_index[term].append({
                "flower_key": flower["key"],
                "category":   category,
            })

    # ── 5. TF-IDF term-document matrix ───────────────────────────────────────
    # analyzer=_tokenize_and_stem keeps corpus and query vocab aligned.
    # sublinear_tf=True applies 1 + log(tf) to dampen very frequent repeats.
    # ngram_range=(1, 2) captures phrases like 'low maintenance', 'everlasting love'.
    documents = [_build_document(f) for f in flowers]
    vectorizer = TfidfVectorizer(
        analyzer=_tokenize_and_stem,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        dtype=np.float32,
    )
    td_matrix = vectorizer.fit_transform(documents)  # sparse (n_flowers, n_terms)

    return (flowers, vectorizer, td_matrix, keyword_index, known_colors, known_maint, stem_to_word)


# ---------------------------------------------------------------------------
# Rocchio pseudo-relevance feedback
# ---------------------------------------------------------------------------


def _rocchio_expand_tfidf(query_vec, td_matrix, top_indices):
    k = min(ROCCHIO_TOP_K, len(top_indices))
    if k == 0:
        return query_vec

    relevant = td_matrix[top_indices[:k]]
    centroid = relevant.mean(axis=0)

    expanded = ROCCHIO_ALPHA * query_vec + ROCCHIO_BETA * centroid
    return expanded if hasattr(expanded, "toarray") else np.asarray(expanded)


# ---------------------------------------------------------------------------
# Query keyword extraction  (inverted index -- reliable, readable display)
# ---------------------------------------------------------------------------

def _extract_query_keywords(
    query: str,
    keyword_index: dict,
    known_colors: list,
    known_maint: list,
) -> list:
    """
    Extract and categorize query terms using the inverted index.

    - Longest-match-first traversal (postings-merge style):
      'low maintenance' is found before 'low' and 'maintenance' separately.
    - Rarity adjustment 1/sqrt(match_count) naturally implements IDF:
      terms matching fewer flowers score higher.
    - MED fuzzy fallback for near-miss color/maintenance tokens.
    """
    normalized   = _normalize(query)
    query_tokens = set(_tokenize(query))
    seen_terms:  set  = set()
    keywords:    list = []

    for term in sorted(
        keyword_index.keys(),
        key=lambda v: (-v.count(" "), -len(v), v),
    ):
        if term in seen_terms:
            continue
        in_query = (
            f" {term} " in f" {normalized} "
            if " " in term
            else term in query_tokens
        )
        if not in_query:
            continue

        seen_terms.add(term)
        for tok in term.split():
            seen_terms.add(tok)

        postings = keyword_index[term]
        cat_key  = postings[0]["category"]
        rarity   = 1 / math.sqrt(len(postings))
        score    = round(BASE_WEIGHTS.get(cat_key, 1.0) * rarity, 2)

        keywords.append({
            "keyword":  term,
            "category": CATEGORY_LABELS.get(cat_key, "text"),
            "score":    score,
        })

    # MED fuzzy fallback for unmatched tokens
    matched_tokens = {tok for kw in keywords for tok in kw["keyword"].split()}
    for tok in query_tokens - matched_tokens:
        if tok in STOPWORDS or len(tok) <= 2:
            continue
        color_hit = _fuzzy_match(tok, known_colors)
        if color_hit:
            keywords.append({"keyword": color_hit, "category": "color", "score": 1.0})
            continue
        maint_hit = _fuzzy_match(tok, known_maint)
        if maint_hit:
            keywords.append({"keyword": maint_hit, "category": "maintenance", "score": 1.0})

    return sorted(keywords, key=lambda x: (-x["score"], x["keyword"]))[:MAX_QUERY_KEYWORDS]


# ---------------------------------------------------------------------------
# Matched keyword chips  (Element-wise product + stem -> word display)
# ---------------------------------------------------------------------------

def _extract_matched_keywords(
    flower: dict,
    query_tfidf,
    flower_tfidf_row,
    vectorizer,
    stem_to_word: dict,
) -> list:
    """
    Element-wise product of the query TF-IDF vector and the
    flower's TF-IDF row. Non-zero entries = stems in both = lexical matches.
    Score = joint weight (how much this term drove this flower's rank).
    Stems are mapped back to original words for readable display.
    """
    feature_names = vectorizer.get_feature_names_out()
    q_weights = query_tfidf.toarray()[0]
    f_weights = flower_tfidf_row.toarray()[0]

    joint   = q_weights * f_weights
    top_idx = np.argsort(joint)[::-1][:MAX_MATCHED_CHIPS]

    matched = []
    for idx in top_idx:
        if joint[idx] <= 0:
            break
        stem = feature_names[idx]
        if " " in stem:
            display = " ".join(stem_to_word.get(s, s) for s in stem.split())
        else:
            display = stem_to_word.get(stem, stem)

        matched.append({
            "keyword":  display,
            "category": _categorize_chip(stem, flower),
            "score":    round(float(joint[idx]) * 10, 2),
        })
    return matched


def _categorize_chip(stem: str, flower: dict) -> str:
    """
    Category for a matched chip -- stemmed comparison against this
    specific flower's own fields.
    Priority: color > maintenance > plant type > occasion > meaning > name
    """
    color_stems = {s for c in flower["colors"] for s in _tokenize_and_stem(_normalize(c))}
    maint_stems = {s for m in flower["maintenance"] for s in _tokenize_and_stem(_normalize(m))}
    pt_stems = {s for pt in flower["plant_types"] for s in _tokenize_and_stem(_normalize(pt))}
    occ_stems = {s for o in flower["occasions"] for s in _tokenize_and_stem(_normalize(o))}
    meaning_stems = {s for m in flower["meanings"] for s in _tokenize_and_stem(_normalize(m))}
    name_stems = set(_tokenize_and_stem(_normalize(flower["name"])))

    for part in stem.split():
        if part in color_stems: return "color"
        if part in maint_stems: return "maintenance"
        if part in pt_stems: return "plant_type"
        if part in occ_stems: return "occasion"
        if part in meaning_stems: return "meaning"
        if part in name_stems: return "name"

    return "text"



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend_flowers_tfidf(query: str, limit: int = 5) -> dict:
    """
    Full retrieval pipeline for one query.

    1. Vectorize the normalized query with the same analyzer as the corpus.
    2. Rank by cosine similarity, apply Rocchio pseudo-relevance feedback
       using the top-k results, then re-rank on the expanded query vector.
    3. Extract inverted-index keywords for the query breakdown panel.
    4. For each suggestion, build matched chips from the element-wise TF-IDF
       product, mapping stems to display words.
    5. Return full meaning and occasion strings from the dataset; the frontend
       truncates long text with expand controls.
    """
    if not query or not query.strip():
        return {
            "query": query,
            "keywords_used": [],
            "query_latent_radar_chart": None,
            "suggestions": []
        }

    (flowers, vectorizer, td_matrix, keyword_index,
     known_colors, known_maint, stem_to_word) = _load_model()

    query_norm = _normalize(query)
    query_tfidf = vectorizer.transform([query_norm])

    # TF-IDF ranking
    sims_initial = _rank_tfidf(query_tfidf, td_matrix)
    initial_top = np.argsort(sims_initial)[::-1]

    # Rocchio
    expanded_vec = _rocchio_expand_tfidf(query_tfidf, td_matrix, initial_top)
    sims = _rank_tfidf(expanded_vec, td_matrix)

    # Query breakdown
    keywords_used = _extract_query_keywords(
        query, keyword_index, known_colors, known_maint
    )

    top_indices = np.argsort(sims)[::-1]

    suggestions = []
    for idx in top_indices:
        if len(suggestions) >= limit:
            break
        if sims[idx] <= 0:
            break

        flower = flowers[idx]

        cosine_score = float(sims[idx]) * 100
        final_score = round(cosine_score, 2)

        matched = _extract_matched_keywords(
            flower, query_tfidf, td_matrix[idx], vectorizer, stem_to_word
        )

        suggestions.append({
            "name": flower["name"],
            "scientific_name": flower["scientific_name"],
            "colors": flower["colors"],
            "plant_types": flower["plant_types"],
            "maintenance": flower["maintenance"],
            "meanings": flower["meanings"],
            "occasions": flower["occasions"],
            "ir_summary": " ".join(flower["meanings"][:2]).strip(),
            "ir_summary_source": "csv" if flower["meanings"] else "",
            "query_fit_occasion_summary": " ".join(flower["occasions"][:2]).strip(),
            "occasion_summary_source": "csv" if flower["occasions"] else "",
            "image_url": resolve_flower_image_url(flower),
            "score": final_score,
            "matched_keywords": matched,
            "latent_radar_chart": None
        })

    return {
        "query": query,
        "keywords_used": keywords_used,
        "query_latent_radar_chart": None,
        "suggestions": suggestions
    }
