"""
FloraSense retrieval system  -  Inverted Index + TF-IDF + Rocchio
    WIP: Very SLOP right now
Dataset: merged.csv  (name, scientific_name, color, planttype,
                       maintenance, meaning, Special Occasions)

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
- Jaccard similarity         : secondary tiebreaker on token overlap
- Minimum Edit Distance      : Wagner-Fisher DP for fuzzy color/maintenance
                               typo handling

Design principle
----------------
Inverted index  ->  display and keyword categorization  (exact, readable)
TF-IDF + LSA    ->  retrieval ranking                   (semantic, robust)
Full prose      ->  TF-IDF corpus only, never shown to user directly
Display fields  ->  short keyword labels
These layers never interfere with each other.
"""

import csv
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import json
import os

# TODO: Check dependencies remove unnecessary, also mirror changes to requirements.txt
import numpy as np
from nltk.stem import PorterStemmer
#from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).resolve().parent / "data" / "merged.csv"
FLOWER_IMAGE_DIR = Path(__file__).resolve().parent.parent / "data_scraping" / "flower_images"

_IMAGE_MAP_PATH = Path(__file__).resolve().parent.parent / "data_scraping" / "image_map.json"
try:
    _IMAGE_MAP = json.loads(_IMAGE_MAP_PATH.read_text(encoding="utf-8"))
except Exception:
    _IMAGE_MAP = {}

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

_IMAGE_ALIAS_TARGETS = {
    "arum lily": "calla lily",
    "pelargonium": "geranium",
    "zantedeschia": "calla lily",
}

_IMAGE_GENERIC_TOKENS = {"flower", "flowers", "meaning", "meanings"}



# Build filename lookup ONCE
_LOCAL_FLOWER_IMAGE_FILES = {}
_LOCAL_IMAGE_FILENAMES = set()

for file in FLOWER_IMAGE_DIR.iterdir():
    if file.suffix.lower() in _IMAGE_EXTENSIONS:
        key = file.stem.lower()  # e.g. "calla-lily-meaning"
        _LOCAL_FLOWER_IMAGE_FILES[key] = file.name
        _LOCAL_IMAGE_FILENAMES.add(file.name)


_COLOR_FALLBACK_IMAGE_FILENAMES = {
    "blue": "blue-flowers-meaning.jpg",
    "pink": "pink-flowers-meaning.jpg",
    "purple": "purple-flowers-meaning.jpg",
    "white": "white-flowers.jpg",
    "yellow": "yellow-flowers-meaning.jpg",
}

_GENERIC_FLOWER_IMAGE_FILENAME = "10-most-beautiful-flowers.jpg"

# Number of latent dimensions for SVD / LSA.
# Most of our flower data lives in far fewer than 40 dimensions (small corpus),
# but 40 is a safe upper bound that we cap dynamically below.
# N_COMPONENTS = 40

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

# Wagner-Fisher MED: max edit distance allowed for fuzzy matching
MED_THRESHOLD = 2


# Max label length: meaning entries shorter than this are clean keyword
# labels (e.g. "Everlasting love") rather than scraped prose paragraphs
MAX_LABEL_LEN = 80

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
    Normalize -> tokenize -> remove stopwords -> stem.
    Passed as analyzer to TfidfVectorizer so the corpus and every query
    go through identical processing. This is the same approach used in
    the in-class demo: build the td_matrix with a consistent tokenizer,
    then project queries through the same vocabulary.
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
    Split semicolon-separated meaning entries.
    "Deception; Graciousness; snapdragons possess two meanings..."
    Each chunk becomes a separate entry -- short labels AND long prose
    both contribute vocabulary to TF-IDF / LSA.
    """
    return [p.strip() for p in value.split(";") if p.strip()]


def _extract_display_meanings(value: str) -> list:
    """
    Extract only the short keyword labels from a meaning string for display.
    Discards long prose chunks that are scraped web content unsuitable for UI.

    "Deception; Graciousness; snapdragons possess two meanings..."
    -> ["Deception", "Graciousness"]

    The MAX_LABEL_LEN threshold captures multi-word labels like
    "Tears of the Virgin Mary" while dropping prose paragraphs.
    If no short labels exist (all entries are prose), fall back to
    the first entry truncated to MAX_LABEL_LEN characters.
    """
    parts = [p.strip() for p in value.split(";") if p.strip()]
    short = [p for p in parts if len(p) <= MAX_LABEL_LEN]
    if short:
        return short
    # No short labels --> take the first clause of the prose up to
    # the first period or comma rather than cutting mid-word
    first = parts[0] if parts else ""
    boundary = re.search(r"[.,]", first)
    if boundary:
        return [first[:boundary.start()].strip()]
    return [first] if first else []


def _extract_display_occasions(value: str) -> list:
    """
    Occasions text is free prose. Take only the first clause (up to the
    first colon or period) which typically lists the concrete occasions
    without the surrounding filler sentences.

    "try giving the gift of hydrangeas for: weddings, engagements..."
    -> ["try giving the gift of hydrangeas for: weddings, engagements"]
    """
    if not value.strip():
        return []
    # Split on period or newline, keep first non-empty chunk
    first = re.split(r"\.\s+|\n", value.strip())[0].strip()
    return [first] if first else []


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
# Images
# ---------------------------------------------------------------------------


def _image_match_score(candidate: str, local_key: str) -> int:
    candidate_tokens = [token for token in _tokenize(candidate) if token not in _IMAGE_GENERIC_TOKENS]
    local_tokens = [token for token in _tokenize(local_key) if token not in _IMAGE_GENERIC_TOKENS]
    if not candidate_tokens or not local_tokens:
        return -1

    candidate_token_set = set(candidate_tokens)
    local_token_set = set(local_tokens)
    overlap = candidate_token_set & local_token_set
    if not overlap:
        return -1

    score = len(overlap) * 10
    if local_key.startswith(candidate):
        score += 20
    elif candidate.startswith(local_key):
        score += 12
    elif candidate in local_key or local_key in candidate:
        score += 6

    if candidate_token_set <= local_token_set:
        score += 16 - max(0, len(local_token_set) - len(candidate_token_set))
    elif local_token_set <= candidate_token_set:
        score += 12 - max(0, len(candidate_token_set) - len(local_token_set))

    return score

def _best_local_image_filename(candidate: str) -> str | None:
    normalized_candidate = _normalize_image_key(candidate)
    if not normalized_candidate:
        return None

    candidates = [normalized_candidate]
    alias_target = _IMAGE_ALIAS_TARGETS.get(normalized_candidate)
    if alias_target:
        candidates.append(_normalize_image_key(alias_target))

    for current_candidate in candidates:
        direct_match = _LOCAL_FLOWER_IMAGE_FILES.get(current_candidate)
        if direct_match:
            return direct_match

    best_score = -1
    best_filename = None
    for current_candidate in candidates:
        for local_key, filename in _LOCAL_FLOWER_IMAGE_FILES.items():
            score = _image_match_score(current_candidate, local_key)
            if score > best_score:
                best_score = score
                best_filename = filename

    return best_filename if best_score >= 20 else None

def _normalize_image_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def _matching_image_candidates(flower: dict):
    candidates = []

    if flower.get("name"):
        candidates.append(("name", flower["name"]))

    if flower.get("scientific_name"):
        candidates.append(("scientific", flower["scientific_name"]))

    for pt in flower.get("plant_types", []):
        candidates.append(("plant_type", pt))

    return [(label, _normalize_image_key(c)) for label, c in candidates]


def _resolve_flower_image_url(flower: dict) -> str | None:
    for _, candidate in _matching_image_candidates(flower):
        filename = _best_local_image_filename(candidate)
        if filename:
            return f"/api/flower-images/{filename}"

        for remote_key, remote_url in _IMAGE_MAP.items():
            normalized_remote_key = _normalize_image_key(remote_key)
            if candidate == normalized_remote_key or candidate in normalized_remote_key or normalized_remote_key in candidate:
                return remote_url

    return _fallback_image_url_from_metadata(flower)

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



# ---------------------------------------------------------------------------
# Minimum Edit Distance  (Wagner-Fisher algorithm)
# ---------------------------------------------------------------------------

def _med(s: str, t: str) -> int:
    """
    Wagner-Fisher dynamic programming algorithm for Minimum Edit Distance.

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
# Jaccard similarity
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    """
    Jaccard similarity = |A intersect B| / |A union B|

    Used as a secondary tiebreaker after cosine ranking.
    Flowers whose stemmed token set overlaps more with the query
    get a small bonus, resolving close cosine scores.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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
    maximum vocabulary coverage for TF-IDF ranking. The display layer
    uses the separate display_meanings / display_occasions fields.
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
    Builds and caches everything needed for retrieval.
    TF-IDF term-document matrix. 
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
                    "meanings":         set(),   # full text -- for TF-IDF only
                    "display_meanings": set(),   # short labels -- for UI
                    "occasions":        set(),   # full text -- for TF-IDF only
                    "display_occasions":set(),   # first clause -- for UI
                }
            e = grouped[key]
            if row.get("color", "").strip():
                e["colors"].add(row["color"].strip())
            if row.get("maintenance", "").strip():
                e["maintenance"].add(row["maintenance"].strip())
            for pt in _split_csv_cell(row.get("planttype", "")):
                e["plant_types"].add(pt)

            raw_meaning = row.get("meaning", "")
            for m in _split_meanings(raw_meaning):
                e["meanings"].add(m)
            for m in _extract_display_meanings(raw_meaning):
                e["display_meanings"].add(m)

            occ = row.get("Special Occasions", "").strip()
            if occ:
                e["occasions"].add(occ)
                for d in _extract_display_occasions(occ):
                    e["display_occasions"].add(d)

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
            "display_meanings":  sorted(e["display_meanings"]),
            "occasions":         sorted(e["occasions"]),
            "display_occasions": sorted(e["display_occasions"]),
        }
        for e in grouped.values()
    ]

    # ── 2. Known field values for MED fuzzy matching ─────────────────────────
    known_colors = list({c for f in flowers for c in f["colors"]})
    known_maint  = list({m for f in flowers for m in f["maintenance"]})

    # ── 3. Stem -> display word map ──────────────────────────────────────────
    # First occurrence of each stem wins. Lets matched keyword chips show
    # the original word ('everlasting') rather than the raw stem ('everlast').
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
    # analyzer=_tokenize_and_stem ensures the corpus and queries use identical
    # processing -- the same vocabulary the notebook builds with fit_transform.
    # sublinear_tf=True applies 1 + log(tf) to dampen repeated tokens.
    # ngram_range=(1,2) captures phrases like 'low maintenance', 'everlasting love'.
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
# Matched keyword chips  (Hadamard product + stem -> word display)
# ---------------------------------------------------------------------------

def _extract_matched_keywords(
    flower: dict,
    query_tfidf,
    flower_tfidf_row,
    vectorizer,
    stem_to_word: dict,
) -> list:
    """
    Element-wise (Hadamard) product of the query TF-IDF vector and the
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
    Full pipeline per request
    -------------------------
    1. Normalize query -> TF-IDF vector  (same vocab as corpus)
       NO SVD
    3. Initial cosine ranking
    4. Rocchio expansion: shift query toward centroid of top-3 results
    5. Re-rank with expanded query vector
    6. Jaccard tiebreaker for near-equal cosine scores
    7. Inverted index -> query breakdown keywords 
    8. Hadamard product -> per-card matched keyword chips (stem -> word map)
    9. Display meanings: short keyword labels from dataset
   10. Display occasions: first clause of occasion prose
   11. Chunking fallback: if a display field has long text remaining,
       slide a window to extract the most query-relevant passage
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
    query_stems = set(_tokenize_and_stem(query_norm))

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
    #top_score = float(sims[top_indices[0]]) if len(top_indices) else 1.0

    suggestions = []
    for idx in top_indices:
        if len(suggestions) >= limit:
            break
        if sims[idx] <= 0:
            break

        flower = flowers[idx]

        flower_stems = set(_tokenize_and_stem(" ".join([
            flower["name"], *flower["colors"], *flower["maintenance"],
            *flower["plant_types"], *flower["meanings"], *flower["occasions"],
        ])))

        jaccard_bonus = _jaccard(query_stems, flower_stems) * 5

        cosine_score = float(sims[idx]) * 100
        final_score = round(cosine_score, 2) # no jaccard bonus if we want to be consistent with raw sim score

        matched = _extract_matched_keywords(
            flower, query_tfidf, td_matrix[idx], vectorizer, stem_to_word
        )

        display_meanings = flower["display_meanings"]

        display_occasions = flower["display_occasions"]

        suggestions.append({
            "name": flower["name"],
            "scientific_name": flower["scientific_name"],
            "colors": flower["colors"],
            "plant_types": flower["plant_types"],
            "maintenance": flower["maintenance"],
            "meanings": display_meanings,
            "occasions": display_occasions,
            "image_url": _resolve_flower_image_url(flower),
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
