"""
FloraSense retrieval system  -  Inverted Index + TF-IDF + LSA + Rocchio
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
- Vector space model         : each flower and query is a vector in
                               term space, then compressed into LSA space
- TruncatedSVD / LSA         : M = U * S * V^T  decomposes the term-doc
                               matrix into latent dimensions. Based
                               directly on the in-class SVD demo:
                               query projected as  q * V  (words_compressed)
                               then ranked by cosine similarity against
                               U * S  (docs_compressed_normed)
- Cosine similarity          : l2-normalized dot product for ranking
- Rocchio's method           : pseudo-relevance feedback shifts query
                               vector toward centroid of top-k results
- Jaccard similarity         : secondary tiebreaker on token overlap
- Minimum Edit Distance      : Wagner-Fisher DP for fuzzy color/maintenance
                               typo handling
- Chunking                   : sliding window over meaning/occasion prose
                               to extract the most query-relevant passage
                               for display, without relying on punctuation

Design principle
----------------
Inverted index  ->  display and keyword categorization  (exact, readable)
TF-IDF + LSA    ->  retrieval ranking                   (semantic, robust)
Full prose      ->  TF-IDF corpus only, never shown to user directly
Display fields  ->  short keyword labels + best sliding-window chunk
These layers never interfere with each other.
"""

import csv
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).resolve().parent / "data" / "merged.csv"

# Number of latent dimensions for SVD / LSA.
# Most of our flower data lives in far fewer than 40 dimensions (small corpus),
# but 40 is a safe upper bound that we cap dynamically below.
N_COMPONENTS = 40

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

# Sliding window chunking parameters for prose display
CHUNK_SIZE = 25   # tokens per window
CHUNK_STEP = 8    # tokens to advance per step

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
    maximum vocabulary coverage in the LSA space. The display layer
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
# Chunking  (sliding window for prose display)
# ---------------------------------------------------------------------------

def _best_chunk(prose: str, query_stems: set) -> str:
    """
    Sliding window chunking over prose tokens.

    Slides a window of CHUNK_SIZE tokens across the prose, stepping
    CHUNK_STEP tokens at a time. Each window is scored by how many of
    its stems overlap with the query stems. The highest-scoring window
    is returned as a readable phrase.

    This avoids relying on sentence-boundary punctuation entirely, which
    is important because much of the meaning text in merged.csv is
    run-on prose scraped from web pages with inconsistent punctuation.

    If the prose is shorter than CHUNK_SIZE tokens it is returned as-is.
    """
    tokens = _tokenize(prose)
    stems  = [_stemmer.stem(t) for t in tokens]

    if len(tokens) <= CHUNK_SIZE:
        return prose

    best_score        = -1
    best_chunk_tokens = tokens[:CHUNK_SIZE]

    for start in range(0, len(tokens) - CHUNK_SIZE + 1, CHUNK_STEP):
        window_stems = set(stems[start : start + CHUNK_SIZE])
        overlap      = len(window_stems & query_stems)
        if overlap > best_score:
            best_score        = overlap
            best_chunk_tokens = tokens[start : start + CHUNK_SIZE]

    # Snap to nearest clause boundary in the original prose so we
    # don't return "...friendship that is still in" mid-sentence.
    # Find where the best chunk starts in the original prose, then
    # extend to the next punctuation mark.
    chunk_start_word = best_chunk_tokens[0]
    # Find the character position of the chunk start in the prose
    pattern = r"\b" + re.escape(chunk_start_word) + r"\b"
    match = re.search(pattern, prose.lower())
    if not match:
        return " ".join(best_chunk_tokens)

    # From that position, grab text up to the next . , ; or end of string
    excerpt = prose[match.start():]
    boundary = re.search(r"[.,;]", excerpt)
    if boundary:
        return excerpt[:boundary.start()].strip()
    # No boundary found --> return the full remaining text up to a word limit
    words = excerpt.split()
    return " ".join(words[:CHUNK_SIZE]).strip()



# ---------------------------------------------------------------------------
# Model loader  (lru_cache -- built once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model() -> tuple:
    """
    Builds and caches everything needed for retrieval.

    SVD decomposition follows the in-class notebook exactly:
        td_matrix = TF-IDF term-document matrix  (n_flowers x n_terms)
        docs_compressed, s, words_compressed = svds(td_matrix, k=40)
        words_compressed = words_compressed.T   (n_terms x k)
        docs_compressed_normed = normalize(docs_compressed)  (n_flowers x k)

    At query time:
        query_vec = normalize(query_tfidf.dot(words_compressed))  (1 x k)
        sims = docs_compressed_normed.dot(query_vec.T)
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

    # ── 6. SVD decomposition (following in-class notebook exactly) ───────────
    #
    # In-class:   docs_compressed, s, words_compressed = svds(td_matrix, k=40)
    #             words_compressed = words_compressed.T
    #             docs_compressed_normed = normalize(docs_compressed)
    #
    # Here we use TruncatedSVD which does the same decomposition.
    # svd.fit_transform(td_matrix)  returns  U * S  (docs_compressed)
    # svd.components_.T             returns  V       (words_compressed)
    #
    # At query time we project: query_vec = normalize(query_tfidf * V)
    # This is identical to the notebook's:
    #   query_vec = normalize(query_tfidf.dot(words_compressed))
    n_components = min(N_COMPONENTS, td_matrix.shape[1] - 1, len(flowers) - 1)
    svd = TruncatedSVD(n_components=n_components, algorithm="randomized", random_state=42)

    docs_compressed        = svd.fit_transform(td_matrix)    # (n_flowers, k)  = U * S
    words_compressed       = svd.components_.T               # (n_terms, k)    = V
    docs_compressed_normed = normalize(docs_compressed)      # l2 row-normalize

    return (flowers, vectorizer, words_compressed, docs_compressed_normed,
            td_matrix, keyword_index, known_colors, known_maint, stem_to_word)


# ---------------------------------------------------------------------------
# Rocchio pseudo-relevance feedback
# ---------------------------------------------------------------------------

def _rocchio_expand(
    query_vec: np.ndarray,
    docs_compressed_normed: np.ndarray,
    top_indices: np.ndarray,
) -> np.ndarray:
    """
    Rocchio query expansion:

        q_new = alpha * q_original  +  beta * (1/k) * sum(relevant_docs)

    Shifts the query vector toward the centroid of the top-k initial
    results, then re-normalizes. This biases re-ranking toward the
    semantic neighborhood of the best initial matches.
    """
    k = min(ROCCHIO_TOP_K, len(top_indices))
    if k == 0:
        return query_vec
    relevant = docs_compressed_normed[top_indices[:k]]             # (k, n_components)
    centroid = relevant.mean(axis=0, keepdims=True)                # (1, n_components)
    expanded = ROCCHIO_ALPHA * query_vec + ROCCHIO_BETA * centroid
    return normalize(expanded, norm="l2")


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
    color_stems   = {s for c  in flower["colors"]      for s in _tokenize_and_stem(_normalize(c))}
    maint_stems   = {s for m  in flower["maintenance"]  for s in _tokenize_and_stem(_normalize(m))}
    pt_stems      = {s for pt in flower["plant_types"]  for s in _tokenize_and_stem(_normalize(pt))}
    occ_stems     = {s for o  in flower["occasions"]    for s in _tokenize_and_stem(_normalize(o))}
    meaning_stems = {s for m  in flower["meanings"]     for s in _tokenize_and_stem(_normalize(m))}
    name_stems    = set(_tokenize_and_stem(_normalize(flower["name"])))

    for part in stem.split():
        if part in color_stems:    return "color"
        if part in maint_stems:    return "maintenance"
        if part in pt_stems:       return "plant type"
        if part in occ_stems:      return "occasion"
        if part in meaning_stems:  return "meaning"
        if part in name_stems:     return "name"
    return "text"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend_flowers(query: str, limit: int = 5) -> dict:
    """
    Full pipeline per request
    -------------------------
    1. Normalize query -> TF-IDF vector  (same vocab as corpus)
    2. Project into LSA space:  query_vec = normalize(query_tfidf * V)
       (mirrors the in-class notebook: query_vec = normalize(query_tfidf.dot(words_compressed)))
    3. Initial cosine ranking:  sims = docs_compressed_normed.dot(query_vec.T)
    4. Rocchio expansion: shift query toward centroid of top-3 results
    5. Re-rank with expanded query vector
    6. Jaccard tiebreaker for near-equal cosine scores
    7. Inverted index -> query breakdown keywords (readable, correctly categorized)
    8. Hadamard product -> per-card matched keyword chips (stem -> word map)
    9. Display meanings: short keyword labels from dataset
   10. Display occasions: first clause of occasion prose
   11. Chunking fallback: if a display field has long text remaining,
       slide a window to extract the most query-relevant passage
    """
    if not query or not query.strip():
        return {"query": query, "keywords_used": [], "suggestions": []}

    (flowers, vectorizer, words_compressed, docs_compressed_normed,
     td_matrix, keyword_index, known_colors, known_maint, stem_to_word) = _load_model()

    # Transform query into TF-IDF space -- same vocabulary as corpus
    query_tfidf = vectorizer.transform([_normalize(query)])          # sparse (1, n_terms)
    query_stems = set(_tokenize_and_stem(_normalize(query)))

    # Project query into LSA space using V matrix (words_compressed)
    # This is identical to the notebook:
    #   query_vec = normalize(query_tfidf.dot(words_compressed)).squeeze()
    query_vec = normalize(query_tfidf.toarray().dot(words_compressed))  # (1, k)

    # Initial cosine ranking
    # sims = docs_compressed_normed.dot(query_vec.T) -- same as notebook
    sims_initial = docs_compressed_normed.dot(query_vec.T).squeeze()   # (n_flowers,)
    initial_top  = np.argsort(sims_initial)[::-1]

    # Rocchio expansion -> re-rank
    expanded_vec  = _rocchio_expand(query_vec, docs_compressed_normed, initial_top)
    sims_expanded = docs_compressed_normed.dot(expanded_vec.T).squeeze()

    # Query breakdown via inverted index
    keywords_used = _extract_query_keywords(
        query, keyword_index, known_colors, known_maint
    )

    top_indices = np.argsort(sims_expanded)[::-1]
    top_score   = float(sims_expanded[top_indices[0]]) if len(top_indices) else 1.0

    suggestions = []
    for idx in top_indices:
        if len(suggestions) >= limit:
            break
        if sims_expanded[idx] <= 0:
            break

        flower = flowers[idx]

        # Jaccard tiebreaker -- small bonus (max +5 pts)
        flower_stems = set(_tokenize_and_stem(" ".join([
            flower["name"], *flower["colors"], *flower["maintenance"],
            *flower["plant_types"], *flower["meanings"], *flower["occasions"],
        ])))
        jaccard_bonus = _jaccard(query_stems, flower_stems) * 5

        cosine_score = (float(sims_expanded[idx]) / top_score) * 100 if top_score > 0 else 0.0
        final_score  = round(min(cosine_score + jaccard_bonus, 100.0), 2)

        matched = _extract_matched_keywords(
            flower, query_tfidf, td_matrix[idx], vectorizer, stem_to_word
        )

        # Display meanings: use short keyword labels.
        # Apply chunking only if a label is still too long (shouldn't happen
        # often after _extract_display_meanings, but handles edge cases).
        display_meanings = [
            m if len(m) <= MAX_LABEL_LEN else _best_chunk(m, query_stems)
            for m in flower["display_meanings"]
        ]

        # Display occasions: already trimmed to first clause.
        # Apply chunking if the clause is still long.
        display_occasions = [
            o if len(_tokenize(o)) <= CHUNK_SIZE else _best_chunk(o, query_stems)
            for o in flower["display_occasions"]
        ]

        suggestions.append({
            "name":             flower["name"],
            "scientific_name":  flower["scientific_name"],
            "colors":           flower["colors"],
            "plant_types":      flower["plant_types"],
            "maintenance":      flower["maintenance"],
            "meanings":         display_meanings,
            "occasions":        display_occasions,
            "score":            final_score,
            "matched_keywords": matched,
        })

    return {
        "query":         query,
        "keywords_used": keywords_used,
        "suggestions":   suggestions,
    }