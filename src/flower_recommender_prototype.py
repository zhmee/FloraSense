"""
FloraSense retrieval system  -  TF-IDF + LSA
Dataset: merged.csv  
    (common_name, scientific_name, color, planttype,
                    maintenance, meaning, occasions)

Pipeline
--------
1. Load merged.csv and group multi row flowers into single documents
2. Build weighted document strings
3. Fit a TfidfVectorizer with (1,2)-gram range, sublinear TF, and a stemming
   tokenizer so things like "loveliness" -> "love" still match
4. Compress with TruncatedSVD (Latent Semantic Analysis) to capture latent dimensions
5. At query time: transform the query -> project into LSA space -> cosine similarity score (scaled)
6. Return top k results in the JSON shape the frontend expects
"""

import csv
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# -------------
# Config
# --------------

DATA_FILE = Path(__file__).resolve().parent / "data" / "merged.csv"

# Latent dimensions for TruncatedSVD
N_COMPONENTS = 50

# Token repetition per field for TF-IDF
FIELD_REPEAT = {
    "name":            3,   # exact flower name is the most important
    "meaning":         3,   # meaning also as important
    "color":           2,   # color should match
    "maintenance":     2,   # maintenance should match
    "occasions":       2,   # TODO: if we want occasion as much as meaning (later on)
    "plant_type":      1,
    "scientific_name": 1,
}

#TODO: MORE stopwords
STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "best", "by", "for", "from", "i",
    "in", "is", "it", "like", "me", "my", "myself", "of", "or", "our",
    "ours", "show", "something", "that", "the", "to", "want", "we", "with",
}

MAX_MATCHED_CHIPS  = 8   # keyword chips shown per flower card
MAX_QUERY_KEYWORDS = 10  # terms shown in the Query Breakdown

# ---------------------------------------------------------------
# Stemming tokenizer  (for corpus build AND query transform)
# ---------------------------------------------------------------

_stemmer = PorterStemmer()


def _tokenize_and_stem(text: str) -> list[str]:
    """
    Normalize -> tokenize -> remove stopwords -> stem.
    Pass this as `analyzer` to TfidfVectorizer so both corpus documents
    and queries go through identical processing.
    """
    tokens = re.findall(r"[a-z0-9]+", re.sub(r"[^a-z0-9\s]+", " ", text.lower()))
    return [
        _stemmer.stem(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]


# ---------------------
# Text helpers
# ---------------------

def _normalize(text: str) -> str:
    """
    Lowercase and strip punctuation
    """
    return re.sub(r"[^a-z0-9\s]+", " ", text.lower()).strip()


def _split_csv_cell(value: str) -> list[str]:
    """
    Split comma separated field values (our plant types)
    """
    return [p.strip() for p in value.split(",") if p.strip()]


def _split_meanings(value: str) -> list[str]:
    """
    Split meaning entries separated by semicolons.
    """
    return [p.strip() for p in value.split(";") if p.strip()]


def _categorize_term(term: str, flower: dict) -> str:
    """
    Category label for a matched term (used just for keyword chips on the returned flower cards).

    Stemmed before comparison.

    Priority order matches term_category_map build order below:
        color -> maintenance -> plant type -> occasion -> meaning -> name -> text
            
    TODO: Handles priority b/c if "wedding" is queried, it can exist both as a meaning and 
    in occasion, we will mark it as occasion if it appears in occasion first.

        EX: "white flowers for gratitude for wedding anniversary"
            This will return Hortensia as #1, we will label "wed" as occasion though it
            appears in both occasion and meaning.
            But for Freesia #3, will return "wed" as meaning bc it only appears there.

    TODO: Note the difference between this and Query Breakdown labels
            Query Breakdown labels uses the first instance as the label        
    """
    color_stems: set[str] = set()
    for c in flower["colors"]:
        color_stems.update(_tokenize_and_stem(_normalize(c)))

    maintenance_stems: set[str] = set()
    for m in flower["maintenance"]:
        maintenance_stems.update(_tokenize_and_stem(_normalize(m)))

    plant_type_stems: set[str] = set()
    for pt in flower["plant_types"]:
        plant_type_stems.update(_tokenize_and_stem(_normalize(pt)))

    occasion_stems: set[str] = set()
    for occ in flower["occasions"]:
        occasion_stems.update(_tokenize_and_stem(_normalize(occ)))

    meaning_stems: set[str] = set()
    for m in flower["meanings"]:
        meaning_stems.update(_tokenize_and_stem(_normalize(m)))

    name_stems: set[str] = set(_tokenize_and_stem(_normalize(flower["name"])))

    if term in color_stems:
        return "color"
    if term in maintenance_stems:
        return "maintenance"
    if term in plant_type_stems:
        return "plant type"
    if term in occasion_stems:
        return "occasion"
    if term in meaning_stems:
        return "meaning"
    if term in name_stems:
        return "name"
    return "text"


# --------------------
# Document builder
# --------------------

def _build_document(flower: dict) -> str:
    """
    Concatenate all flower fields into one weighted text string.
    Field importance is encoded via token repetition (FIELD_REPEAT), which
    raises raw term frequency and therefore TF-IDF weight for that field.
    The stemming tokenizer runs over this string at vectorizer fit time.
    """
    parts: list[str] = []

    parts += [_normalize(flower["name"])]            * FIELD_REPEAT["name"]
    parts += [_normalize(flower["scientific_name"])] * FIELD_REPEAT["scientific_name"]

    for color in flower["colors"]:
        parts += [_normalize(color)] * FIELD_REPEAT["color"]

    for maint in flower["maintenance"]:
        parts += [_normalize(maint)] * FIELD_REPEAT["maintenance"]

    for meaning in flower["meanings"]:
        parts += [_normalize(meaning)] * FIELD_REPEAT["meaning"]

    for occ in flower["occasions"]:
        parts += [_normalize(occ)] * FIELD_REPEAT["occasions"]

    for pt in flower["plant_types"]:
        parts += [_normalize(pt)] * FIELD_REPEAT["plant_type"]

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Model loader  (lru_cache -> CSV + SVD built only once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model() -> tuple:
    """
    Returns
    -------
    flowers           : list of flower dicts
    vectorizer        : fitted TfidfVectorizer
    svd               : fitted TruncatedSVD
    tfidf_matrix      : sparse  (n_flowers x n_terms)  TF-IDF matrix
    lsa_matrix        : dense   (n_flowers x n_components)  l2-normalised LSA matrix
    term_category_map : dict[stemmed_term -> category_label]
    """

    # Load CSV, group multi color / multi meaning rows per flower
    grouped: dict[str, dict] = {}

    with DATA_FILE.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            common = row.get("common_name", "").strip()
            sci    = row.get("scientific_name", "").strip()
            key    = _normalize(common or sci)
            if not key:
                continue

            if key not in grouped:
                grouped[key] = {
                    "key":             key,
                    "name":            common or sci or "Unknown",
                    "scientific_name": sci or "Unknown",
                    "colors":          set(),
                    "plant_types":     set(),
                    "maintenance":     set(),
                    "meanings":        set(),
                    "occasions":       set(),
                }

            entry = grouped[key]
            if row.get("color", "").strip():
                entry["colors"].add(row["color"].strip())
            if row.get("maintenance", "").strip():
                entry["maintenance"].add(row["maintenance"].strip())
            for pt in _split_csv_cell(row.get("planttype", "")):
                entry["plant_types"].add(pt)
            for m in _split_meanings(row.get("meaning", "")):
                entry["meanings"].add(m)
            # occasions is free prose, store as a single string per flower,
            # deduplicating identical prose that repeats across color rows
            occ_text = row.get("occasions", "").strip()
            if occ_text:
                entry["occasions"].add(occ_text)

    # Convert sets -> sorted lists for deterministic API output says Claude
    flowers = [
        {
            "key":             e["key"],
            "name":            e["name"],
            "scientific_name": e["scientific_name"],
            "colors":          sorted(e["colors"]),
            "plant_types":     sorted(e["plant_types"]),
            "maintenance":     sorted(e["maintenance"]),
            "meanings":        sorted(e["meanings"]),
            "occasions":       sorted(e["occasions"]),
        }
        for e in grouped.values()
    ]

    # Build term -> category map  (priority = setdefault order)
    # setdefault only writes if the key is absent, so the first field to
    # register a stemmed token wins. 
    # Priority:
    #   color > maintenance > plant_type > occasion > meaning > name

    # TODO: This `term_category_map` is the cause of the 
    #          `_categorize_term` difference from before,
    #        _extract_matched_keywords vs _extract_query_keywords

    term_category_map: dict[str, str] = {}

    for flower in flowers:
        for color in flower["colors"]:
            for tok in _tokenize_and_stem(_normalize(color)):
                term_category_map.setdefault(tok, "color")
            phrase = " ".join(_tokenize_and_stem(_normalize(color)))
            if phrase:
                term_category_map.setdefault(phrase, "color")

        for maint in flower["maintenance"]:
            for tok in _tokenize_and_stem(_normalize(maint)):
                term_category_map.setdefault(tok, "maintenance")
            phrase = " ".join(_tokenize_and_stem(_normalize(maint)))
            if phrase:
                term_category_map.setdefault(phrase, "maintenance")

        for pt in flower["plant_types"]:
            for tok in _tokenize_and_stem(_normalize(pt)):
                term_category_map.setdefault(tok, "plant type")
            phrase = " ".join(_tokenize_and_stem(_normalize(pt)))
            if phrase:
                term_category_map.setdefault(phrase, "plant type")

        for occ in flower["occasions"]:
            for tok in _tokenize_and_stem(_normalize(occ)):
                term_category_map.setdefault(tok, "occasion")

        for meaning in flower["meanings"]:
            for tok in _tokenize_and_stem(_normalize(meaning)):
                term_category_map.setdefault(tok, "meaning")
            phrase = " ".join(_tokenize_and_stem(_normalize(meaning)))
            if phrase:
                term_category_map.setdefault(phrase, "meaning")

        for tok in _tokenize_and_stem(_normalize(flower["name"])):
            term_category_map.setdefault(tok, "name")

    documents = [_build_document(f) for f in flowers]

    # TF-IDF vectorization 
    # analyzer = _tokenize_and_stem  
    # sublinear_tf=True  applies 1 + log(tf) to dampen high frequency terms
    # ngram_range=(1,2)  captures "low maintenance", "everlasting love", etc.
    vectorizer = TfidfVectorizer(
        analyzer=_tokenize_and_stem,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)  # sparse (n_flowers, n_terms)

    # Latent Semantic Analysis (TruncatedSVD)
    #   projects flowers into N_COMPONENTS latent dimensions.
    n_components = min(N_COMPONENTS, tfidf_matrix.shape[1] - 1, len(flowers) - 1)
    svd = TruncatedSVD(n_components=n_components, algorithm="randomized", random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)     # dense (n_flowers, n_components)
    lsa_matrix = normalize(lsa_matrix, norm="l2")    # unit norm -> cosine sim == dot product

    return flowers, vectorizer, svd, tfidf_matrix, lsa_matrix, term_category_map


# -------------------------------------------------------
# Keyword extraction helpers  (for frontend)
# -------------------------------------------------------

def _extract_query_keywords(
    query_tfidf,
    vectorizer,
    term_category_map: dict,
) -> list[dict]:
    """
    Pull the top weighted stems from the TF-IDF query vector.
    For: Query Breakdown
    """
    feature_names = vectorizer.get_feature_names_out()
    weights = query_tfidf.toarray()[0]

    top_indices = np.argsort(weights)[::-1][:MAX_QUERY_KEYWORDS]
    keywords = []
    for idx in top_indices:
        if weights[idx] <= 0:
            break
        term = feature_names[idx]
        if term in term_category_map:
            category = term_category_map[term]
        elif " " in term:
            category = "meaning"   # unseen bigrams are almost always meaning phrases
        else:
            category = "text"
        keywords.append({
            "keyword":  term,
            "category": category,
            "score":    round(float(weights[idx]) * 10, 2),
        })
    return keywords


def _extract_matched_keywords(
    flower: dict,
    query_tfidf,
    flower_tfidf_row,
    vectorizer,
) -> list[dict]:
    """
    Hadamard (element-wise) product of the query TF-IDF vector and a flower's
    TF-IDF row. Non-zero entries are stems present in both -> lexical matches.
    Score = joint relevance weight.
    """
    feature_names = vectorizer.get_feature_names_out()
    q_weights = query_tfidf.toarray()[0]
    f_weights = flower_tfidf_row.toarray()[0]

    joint = q_weights * f_weights
    top_indices = np.argsort(joint)[::-1][:MAX_MATCHED_CHIPS]

    matched = []
    for idx in top_indices:
        if joint[idx] <= 0:
            break
        term = feature_names[idx]
        matched.append({
            "keyword":  term,
            "category": _categorize_term(term, flower),
            "score":    round(float(joint[idx]) * 10, 2),
        })
    return matched


# for each meaning sentence, just check overlap:
def _filter_prose_by_stems(prose: str, query_stems: set[str]) -> str:
    """
    Return the most query relevant sentences from a prose string.
    Scores each sentence by how many stemmed query tokens it contains,
    then returns the top 2 in their original order.
    """
    sentences = re.split(r'(?<=[.!?])\s+', prose.strip())
    if len(sentences) == 1:
        return prose

    scored = []
    for i, sentence in enumerate(sentences):
        stems = set(_tokenize_and_stem(sentence))
        overlap = len(stems & query_stems)
        scored.append((overlap, i, sentence))

    scored.sort(key=lambda x: (-x[0], x[1]))
    top = sorted(scored[:2], key=lambda x: x[1])
    relevant = [s for _, _, s in top if s.strip()]
    return " ".join(relevant) if relevant else sentences[0]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend_flowers(query: str, limit: int = 5) -> dict:
    """
    Main entry point called by routes.py.

    Parameters
    ----------
    query : str   - raw user query string
    limit : int   - maximum suggestions to return (default 5)

    Returns
    -------
    dict matching the RecommendationResponse TypeScript interface:
    {
        query          : str,
        keywords_used  : [{keyword, category, score}, ...],
        suggestions    : [{
            name, scientific_name, colors, plant_types,
            maintenance, meanings, occasions, score, matched_keywords
        }, ...]
    }
    """
    if not query or not query.strip():
        return {"query": query, "keywords_used": [], "suggestions": []}

    flowers, vectorizer, svd, tfidf_matrix, lsa_matrix, term_category_map = _load_model()

    # Transform query into TF-IDF space 
    query_tfidf = vectorizer.transform([_normalize(query)])   # sparse (1, n_terms)
    query_stems = set(_tokenize_and_stem(_normalize(query)))

    # Project into LSA latent space
    query_lsa = svd.transform(query_tfidf)                    # dense (1, n_components)
    query_lsa = normalize(query_lsa, norm="l2")

    # Cosine similarity against all flowers (= dot product after l2 normalisation)
    similarities = cosine_similarity(query_lsa, lsa_matrix)[0]  # (n_flowers,)

    keywords_used = _extract_query_keywords(query_tfidf, vectorizer, term_category_map)

    top_indices = np.argsort(similarities)[::-1]
    top_score   = float(similarities[top_indices[0]]) if len(top_indices) else 1.0

    suggestions = []
    for idx in top_indices:
        if len(suggestions) >= limit:
            break
        if similarities[idx] <= 0:
            break

        flower  = flowers[idx]
        matched = _extract_matched_keywords(
            flower, query_tfidf, tfidf_matrix[idx], vectorizer
        )
        

        # TODO": Do we like this: Normalise so the top result = 100, rest are relative to it.
        normalised_score = round(
            (float(similarities[idx]) / top_score) * 100, 2
        ) if top_score > 0 else 0.0


        suggestions.append({
            "name":             flower["name"],
            "scientific_name":  flower["scientific_name"],
            "colors":           flower["colors"],
            "plant_types":      flower["plant_types"],
            "maintenance":      flower["maintenance"],
            "meanings":  [_filter_prose_by_stems(m, query_stems) for m in flower["meanings"]],
            "occasions": [_filter_prose_by_stems(o, query_stems) for o in flower["occasions"]],
            "score":            normalised_score,
            "matched_keywords": matched,
        })

    return {
        "query":         query,
        "keywords_used": keywords_used,
        "suggestions":   suggestions,
    }