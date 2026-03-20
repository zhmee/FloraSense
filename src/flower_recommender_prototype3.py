"""
FloraSense prototype 3: pure latent retrieval with TF-IDF + SVD.

This file builds one text document for each flower, turns those documents into
numeric vectors, reduces them with SVD, and then finds the flowers whose vectors
are closest to the user's query.

`merged.csv` is only used to add more descriptive text to each flower document.
It is not used as a direct lookup table and it does not add hand-written scoring
rules at query time.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# the file locations
DATA_FILE = Path(__file__).resolve().parent / "data" / "merged.csv"
TEXT_CORPUS_DIR = Path(__file__).resolve().parent.parent / "data_scraping" / "flower_texts"

# configuration values for the model and for the returned UI payload.
MAX_SVD_COMPONENTS = 96
MAX_QUERY_KEYWORDS = 10
MAX_MATCHED_KEYWORDS = 8
MAX_HIGHLIGHTS = 3
GENERIC_EXPLANATION_TOKENS = {
    "express",
    "expresses",
    "mean",
    "means",
    "meaning",
    "meanings",
    "represent",
    "represents",
    "symbol",
    "symbols",
    "symbolize",
    "symbolizes",
}

# token pattern because we need words for matching/displaying
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
# try to skip "groups of flowers" as we are trying to build a document per flower
THEMATIC_PREFIXES = (
    "10-",
    "chinese-",
    "christmas-",
    "easter-",
    "flower-color-",
    "flower-of-",
    "flowers-",
    "funeral-",
    "i-love-you-",
    "japanese-",
    "june-",
    "language-of-",
    "may-",
    "mothers-day-",
    "pink-",
    "purple-",
    "rare-",
    "sympathy-",
    "thank-you",
    "white-",
    "yellow-",
    "blue-",
)

# START OF TEXT CLEANUP HELPERS

def _normalize(text: str) -> str:
    """
    Convert text to a simple comparable form:
    lowercase, no punctuation, spaces kept.
    """
    return re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower()).strip()


def _tokenize(text: str) -> list[str]:
    """
    Split normalized text into basic word-like tokens.
    """
    return TOKEN_PATTERN.findall(_normalize(text))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """
    Remove repeated values while keeping the first version we saw.
    """
    seen = set()
    deduped = []
    for value in values:
        # normalize for comparison because it wasn't working before... (the smallest diffs were 
        # getting counted as entirely different terms)
        normalized = _normalize(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        # keep the original text for display
        deduped.append(value.strip())
    return deduped


def _split_csv_cell(value: str) -> list[str]:
    """
    Some CSV columns store multiple values in one comma-separated string.
    """
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _split_meaning_chunks(value: str) -> list[str]:
    """
    Split a long meaning string into smaller phrases such as
    "love; devotion; purity" -> ["love", "devotion", "purity"].
    """
    parts = re.split(r"[;\n]+", value or "")
    return [part.strip(" .:") for part in parts if part.strip(" .:")]


def _split_passages(text: str) -> list[str]:
    """
    Break article text into paragraph-like passages.
    These passages can later be shown as explanations.
    """
    passages = []
    for chunk in re.split(r"\n\s*\n", text):
        # merge all the paragraphs into a clean string
        cleaned = " ".join(part.strip() for part in chunk.splitlines() if part.strip())
        # short fragments can just get out (they aren't usually useful)
        if len(cleaned.split()) >= 6:
            passages.append(cleaned)
    return passages


def _extract_title(text: str, fallback: str) -> str:
    """
    Use the first non-empty line as the title.
    If that fails, fall back to the name from the filename.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return fallback


def _slug_to_display_name(slug: str) -> str:
    """
    Convert a filename slug such as "forget-me-not-flower-meaning"
    into a nicer display name such as "Forget Me Not".
    """
    base = slug
    for suffix in ("-flower-meaning", "-flower", "-meaning"):
        # remove any common suffixes used in the scraped filenames.
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.replace("-", " ").title().strip() or "Unknown Flower"


def _alias_variants(name: str) -> set[str]:
    """
    Create a few simple name variants so singular and plural forms
    can still match the same flower. (this took too long might want to check logic as well)
    """
    normalized = _normalize(name)
    if not normalized:
        return set()

    variants = {normalized}
    parts = normalized.split()
    last = parts[-1]

    if len(parts) == 1:
        # plural handling for single-word names
        if last.endswith("y") and len(last) > 1:
            variants.add(last[:-1] + "ies")
        elif not last.endswith("s"):
            variants.add(last + "s")
    else:
        # holy moly multi-word names is DIFFERENT, ended up just checking last word
        if last.endswith("y") and len(last) > 1:
            plural_last = last[:-1] + "ies"
        elif last.endswith("s"):
            plural_last = last
        else:
            plural_last = last + "s"
        variants.add(" ".join([*parts[:-1], plural_last]))

    return {variant for variant in variants if variant}


def _join_human(values: list[str]) -> str:
    """
    Join a list into (kinda) readable English text
    """
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _empty_response(query: str, keywords: list[dict] | None = None) -> dict:
    """
    Common empty response shape used when there is no good result to return.
    """
    return {
        "query": query,
        "keywords_used": keywords or [],
        # there's nothing we can do very sad
        "suggestions": [],
    }


def _new_flower_doc(name: str, scientific_name: str = "Unknown") -> dict:
    """
    Create the base data structure for one flower.
    Later steps keep adding text and metadata to this object.
    """
    display_name = name.strip() or scientific_name.strip() or "Unknown Flower"
    return {
        # `key` is the normalized internal identifier used to match data sources.
        "key": _normalize(display_name),
        "name": display_name,
        "scientific_name": scientific_name.strip() or "Unknown",
        "colors": [],
        "plant_types": [],
        "maintenance": [],
        "meanings": [],
        "occasions": [],
        "aliases": _alias_variants(display_name),
        "article_passages": [],
    }


# ---------------------------------------------------------------------------
# Match different data sources to the same flower and build flower documents
# ---------------------------------------------------------------------------
# BASICALLY, ALL THIS IS JUST TO MATCH THE DIFF DATA SOURCES TO THE SAME FLOWER TO BUILD THE FLOWER DOCUMENTS
def _best_matching_key(name: str, scientific_name: str, docs_by_key: dict[str, dict]) -> str | None:
    """
    Find which existing flower document best matches a CSV row.
    This is needed because the article files and the CSV are not always named
    in exactly the same way (which really pmo man I was wondering what was going on for so long).
    """
    candidate_texts = [text for text in (_normalize(name), _normalize(scientific_name)) if text]
    if not candidate_texts:
        return None

    alias_to_key = {}
    for key, doc in docs_by_key.items():
        for alias in doc["aliases"]:
            # map each alias to the flower document it belongs to
            alias_to_key.setdefault(alias, key)

    for candidate in candidate_texts:
        # SAFEST CASE (PLEASE JUST WORK): alias match
        if candidate in alias_to_key:
            return alias_to_key[candidate]

    best_score = 0
    best_key = None
    for key, doc in docs_by_key.items():
        for alias in doc["aliases"]:
            # ...the safest case didn't work, and very short aliases are too ambiguous to trust
            if len(alias) < 3:
                continue # can't trust it
            alias_tokens = set(alias.split())
            for candidate in candidate_texts:
                score = 0
                if alias in candidate or candidate in alias:
                    # STROOOONNNGG score when one name MOSTLY contains the other
                    score = min(len(alias), len(candidate))
                else:
                    candidate_tokens = set(candidate.split())
                    overlap = len(alias_tokens & candidate_tokens)
                    # yk if it doesn't we gotta look elsewhere for strong overlap
                    if overlap == min(len(alias_tokens), len(candidate_tokens)):
                        score = overlap * 10 + min(len(alias), len(candidate)) / 100
                if score > best_score:
                    # keep the best match bc we loyal like that
                    best_score = score
                    best_key = key

    return best_key if best_score >= 4 else None


def _merge_structured_record(doc: dict, record: dict) -> None:
    """
    Add CSV metadata into an existing flower document.
    This changes the text we later vectorize, but it does not directly score queries bc for
    some reason it didn't work for the past hour.
    """
    if doc["scientific_name"] == "Unknown" and record["scientific_name"] != "Unknown":
        doc["scientific_name"] = record["scientific_name"]

    doc["colors"] = _dedupe_preserve_order(doc["colors"] + record["colors"])
    doc["plant_types"] = _dedupe_preserve_order(doc["plant_types"] + record["plant_types"])
    doc["maintenance"] = _dedupe_preserve_order(doc["maintenance"] + record["maintenance"])
    doc["meanings"] = _dedupe_preserve_order(doc["meanings"] + record["meanings"])
    doc["occasions"] = _dedupe_preserve_order(doc["occasions"] + record["occasions"])
    # you get an alias you get an alias everyone gets an alias (adding more names that can refer to this flower)
    doc["aliases"].update(_alias_variants(record["name"]))
    if record["scientific_name"] != "Unknown":
        doc["aliases"].add(_normalize(record["scientific_name"]))


def _build_structured_passages(flower: dict) -> list[str]:
    """
    Turn structured metadata into short natural-language sentences.
    This lets TF-IDF and SVD learn from the metadata as text (... i think).
    """
    passages = [f"{flower['name']} is a flower."]
    name = flower["name"]

    if flower["scientific_name"] != "Unknown":
        # include the scientific name in sentence form
        passages.append(f"{name} is also known as {flower['scientific_name']}.")
    if flower["plant_types"]:
        # include plant type information
        passages.append(f"{name} is a {_join_human(flower['plant_types'])} plant.")
    for maintenance in flower["maintenance"]:
        # include maintenance information in sentence form
        passages.append(f"{name} is a {maintenance} maintenance flower.")
    if flower["colors"]:
        passages.append(f"{name} flowers come in {_join_human(flower['colors'])}.")

    for color in flower["colors"]:
        # add short color statements and color+maintenance combinations
        # i think they'll help with semantic retrieval without having hand-written rules
        # i was right hehehehe (it's 3 am)
        passages.append(f"{name} can be {color}.")
        for maintenance in flower["maintenance"]:
            passages.append(f"{name} is a {maintenance} maintenance {color} flower.")

    meaning_phrases = []
    for meaning in flower["meanings"]:
        # keep both the full meaning string and smaller phrases from it
        meaning_phrases.extend(_split_meaning_chunks(meaning))
        passages.append(f"{name} meaning: {meaning}")

    short_meaning_phrases = []
    for phrase in _dedupe_preserve_order(meaning_phrases)[:10]:
        # short meaning phrases are easier for retrieval to use (well no sh-)
        if len(phrase.split()) <= 8:
            short_meaning_phrases.append(phrase)
            passages.append(f"{name} symbolizes {phrase}.")

    for color in flower["colors"]:
        for phrase in short_meaning_phrases[:6]:
            # combine color and meaning in one sentence so a query can connect both ideas together
            if len(phrase.split()) <= 5:
                passages.append(f"{name} is a {color} flower that symbolizes {phrase}.")

    for occasion in flower["occasions"]:
        # add occasion text as another clue for retrieval (you see idk if this is really working that well)
        # TODO: FIX OCCASSIONS BC THIS AIN'T WORKING THAT WELL AT ALL.
        passages.append(f"{name} special occasions: {occasion}")

    # was getting duplicates so had to do this before returning
    return _dedupe_preserve_order(passages)


def _has_structured_signal(flower: dict) -> bool:
    # check whether this flower has useful metadata from the CSV (very said if it doesn't)
    return any(
        [
            flower["colors"],
            flower["plant_types"],
            flower["maintenance"],
            flower["meanings"],
            flower["occasions"],
        ]
    )


def _build_document(flower: dict) -> tuple[list[str], str]:
    """"
    WE CAN FINALLY BUILD THE FINAL TEXT DOCUMENT that the model will be able to vectorize and search over
    """
    structured_passages = _build_structured_passages(flower)

    # if ther'es actual metadata, we want that clean text

    if _has_structured_signal(flower):
        passages = structured_passages
    else: # ...yk we can work with the scraped passages otherwise
        passages = _dedupe_preserve_order(structured_passages + flower["article_passages"])

    document_parts = [flower["name"]]
    if flower["scientific_name"] != "Unknown":
        document_parts.append(flower["scientific_name"])
    # put the names first, then the descriptive passages
    document_parts.extend(passages)
    # NOW WE CAN JOIN EVERYTHING INTO ONE TEXT STRING AND FINALLY TURN IT INTO A VECTOR (later tho)
    return passages, "\n\n".join(part for part in document_parts if part)


# HERE'S THE LATER I WAS TALKING ABOUT (turn text into vectors and COMPARE THEM)
def _combined_features(
    texts: list[str],
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
):
    # now we build both word-based and character-based features for the same texts
    # word features help with meaning while character features help with close spellings
    word_matrix = word_vectorizer.transform(texts)
    char_matrix = char_vectorizer.transform(texts)
    # now we do some linalg bc we love linalg
    # (put both feature sets side by side into one combined sparse matrix)
    return hstack([word_matrix, char_matrix], format="csr")


def _fit_lsa(matrix, max_components: int) -> tuple[TruncatedSVD | None, np.ndarray | None]:
    # TRUNCATEDSVD IS THE GOAT TRUST we use ot to reduce the high-dimensional TF-IDF matrix
    # into a smaller latent space where related phrases can end up closer together EVEN IF 
    # THEY ARE NOT EXACT TOKEN MATCHES (now i gotta implement it tho mega sad)
    n_components = min(max_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if n_components < 1:
        # if there are too few rows or features, SVD cannot be fit safely (and can't do it unsafely)
        return None, None

    svd = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        random_state=42,
    )
    # now we must normalize the projected vectors so cosine similarity acts properly (i forgot to do this
    # and my laptop crashed for some reason i don't think they're related but you never know)
    return svd, normalize(svd.fit_transform(matrix), norm="l2")


def _is_generic_flower_term(term: str) -> bool:
    """
    wow, we looking for flowers... how is "flower" helpful man
    """
    tokens = set(_tokenize(term))
    return bool(tokens) and (
        "flower" in tokens
        or "flowers" in tokens
        or tokens <= GENERIC_EXPLANATION_TOKENS
    )


def _is_redundant_keyword(term: str, selected_terms: list[str]) -> bool:
    """
    avoid returning multiple keyword chips that mostly say the same thing
    """
    term_tokens = set(_tokenize(term))
    if not term_tokens:
        return True

    covered_tokens = set()
    for selected in selected_terms:
        selected_tokens = set(_tokenize(selected))
        if not selected_tokens:
            continue
        covered_tokens.update(selected_tokens)
        # skip if the new term is already covered by an earlier chosen term
        if term_tokens <= selected_tokens:
            return True
        # yk skip again if the new term is just a longer version of the same idea
        if selected_tokens <= term_tokens and len(selected_tokens) >= 2:
            return True

    # yk why not skip again if the term is already fully covered by the union of the chosen terms
    return len(term_tokens) >= 2 and term_tokens <= covered_tokens


def _top_feature_terms(row, vectorizer: TfidfVectorizer, limit: int) -> list[dict]:
    """
    Return the highest-weight word features from a sparse row.
    These are only used to explain results to the user (bc they don't wanna know
    all the other features).
    """
    if row.nnz == 0:
        return []

    feature_names = vectorizer.get_feature_names_out()
    dense = row.toarray()[0]
    terms = []
    selected_keywords: list[str] = []

    for index in np.argsort(dense)[::-1]:
        if len(terms) >= limit:
            break
        weight = float(dense[index])
        if weight <= 0:
            # because the weights are sorted descending, everything after this should 
            # theoretically 🤓☝️ also be zero (my bad in the final i'll remove/refine my comments)
            break

        keyword = feature_names[index]
        if _is_generic_flower_term(keyword) or _is_redundant_keyword(keyword, selected_keywords):
            continue

        selected_keywords.append(keyword)
        # this score is only for display. it's not the actual score shown by SVD (we faking it till we making it lets go)
        terms.append(
            {
                "keyword": keyword,
                "category": "semantic",
                "score": round(weight * 10, 2),
            }
        )

    return terms


def _keyword_category_for_flower(keyword: str, flower: dict) -> str:
    """
    Label explanation keywords with a more specific category when they clearly
    match a flower metadata field.
    """
    normalized_keyword = _normalize(keyword)
    keyword_tokens = set(_tokenize(keyword))
    if not normalized_keyword or not keyword_tokens:
        return "semantic"

    category_values = {
        "color": flower["colors"],
        "maintenance": flower["maintenance"],
        "plant_type": flower["plant_types"],
        "meaning": flower["meanings"],
        "occasion": flower["occasions"],
    }

    for category, values in category_values.items():
        for value in values:
            value_tokens = set(_tokenize(value))
            if not value_tokens:
                continue
            if normalized_keyword == _normalize(value):
                return category
            if keyword_tokens <= value_tokens or value_tokens <= keyword_tokens:
                return category

    return "semantic"


def _keyword_category_for_corpus(keyword: str, flowers: list[dict]) -> str:
    """
    Label query breakdown keywords against the corpus as a whole.
    This lets obvious attribute words like "white" show up as "color"
    even before we are talking about one specific flower result.
    """
    category_priority = ("color", "maintenance", "plant_type", "occasion", "meaning")
    category_matches = {category: 0 for category in category_priority}

    for flower in flowers:
        category = _keyword_category_for_flower(keyword, flower)
        if category in category_matches:
            category_matches[category] += 1

    for category in category_priority:
        if category_matches[category] > 0:
            return category

    return "semantic"


def _categorize_feature_terms_for_corpus(terms: list[dict], flowers: list[dict]) -> list[dict]:
    """
    Re-label top query terms with corpus-level metadata categories when possible.
    """
    for term in terms:
        term["category"] = _keyword_category_for_corpus(term["keyword"], flowers)
    return terms


def _select_display_texts(texts: list[str], query: str, limit: int) -> list[str]:
    """
    Choose the most query-relevant meaning or occasion strings to show in the UI.
    """
    deduped = _dedupe_preserve_order(texts)
    if not deduped:
        return []

    # this display-only ranking is based on simple token overlap with the query (IT ISN'T USED TO 
    # ACTUALLY CALCULATE THE SCORE IT'S JUST TO SHOW IN THE UI PROFESSOR DNM PLEASE DON'T BAN ME)
    query_tokens = set(_tokenize(query))
    ranked = sorted(
        deduped,
        # prefer strings with more shared query words, then richer/longer strings
        key=lambda text: (len(set(_tokenize(text)) & query_tokens), len(set(_tokenize(text))), len(text)),
        reverse=True,
    )
    return ranked[:limit]


def _top_passages(
    query: str,
    passages: list[str],
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    svd: TruncatedSVD | None,
) -> list[str]:
    # now we should choose a few relevant passages as fallback explanations (especially when we got 
    # short metadata fields which are missing or not useful)
    if not passages:
        return []

    passage_matrix = _combined_features(passages, word_vectorizer, char_vectorizer)
    query_matrix = _combined_features([query], word_vectorizer, char_vectorizer)

    if svd is None:
        # this the fallback we were talking about: compare the query directly to passage features without SVD.
        similarities = cosine_similarity(query_matrix, passage_matrix)[0]
    else:
        # "normal" case: project query and passages into the same SVD space.
        query_lsa = normalize(svd.transform(query_matrix), norm="l2")
        passage_lsa = normalize(svd.transform(passage_matrix), norm="l2")
        similarities = cosine_similarity(query_lsa, passage_lsa)[0]

    highlights = []
    for index in np.argsort(similarities)[::-1]:
        if len(highlights) >= MAX_HIGHLIGHTS:
            break
        if float(similarities[index]) <= 0:
            break
        # survival of the fittest (keep only the strongest positively related passages)
        highlights.append(passages[index])
    return highlights


# okie not core logic but we have to do loading of the CSV data and scraped flower pages
def _load_csv_records() -> list[dict]:
    # read `merged.csv` and combine repeated rows into one record per flower
    grouped: dict[str, dict] = {}
    with DATA_FILE.open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # use the common name if present, otherwise the scientific name
            common_name = (row.get("name") or "").strip()
            scientific_name = (row.get("scientific_name") or "").strip()
            key = _normalize(common_name or scientific_name)
            if not key:
                # skip USELESS rows (don't identify a flower why you in the dataset).
                continue

            entry = grouped.setdefault(
                key,
                {
                    "name": common_name or scientific_name or "Unknown Flower",
                    "scientific_name": scientific_name or "Unknown",
                    "colors": [],
                    "plant_types": [],
                    "maintenance": [],
                    "meanings": [],
                    "occasions": [],
                },
            )

            # different rows may contribute different pieces of metadata
            if row.get("color", "").strip():
                entry["colors"].append(row["color"].strip())
            if row.get("maintenance", "").strip():
                entry["maintenance"].append(row["maintenance"].strip())
            # `planttype` may contain multiple comma-separated values
            entry["plant_types"].extend(_split_csv_cell(row.get("planttype", "")))

            meaning_value = (row.get("meaning") or "").strip()
            if meaning_value:
                entry["meanings"].append(meaning_value)

            occasion_value = (row.get("Special Occasions") or "").strip()
            if occasion_value:
                entry["occasions"].append(occasion_value)

    return [
        # final deduped record for each flower (bc i had nearly 10 duplicates for one flower like holy moly)
        {
            "name": entry["name"],
            "scientific_name": entry["scientific_name"],
            "colors": _dedupe_preserve_order(entry["colors"]),
            "plant_types": _dedupe_preserve_order(entry["plant_types"]),
            "maintenance": _dedupe_preserve_order(entry["maintenance"]),
            "meanings": _dedupe_preserve_order(entry["meanings"]),
            "occasions": _dedupe_preserve_order(entry["occasions"]),
        }
        for entry in grouped.values()
    ]


def _is_thematic_file(path: Path) -> bool:
    """
    Return True if a file looks like a category/topic page rather than
    a page about one specific flower.
    """
    slug = path.stem
    return slug.endswith("flowers") or slug.startswith(THEMATIC_PREFIXES)


def _build_corpus_docs() -> list[dict]:
    """
    Build the flower documents from scraped text files, then enrich them
    with matching CSV metadata.
    """
    docs_by_key: dict[str, dict] = {}

    for path in sorted(TEXT_CORPUS_DIR.glob("*.txt")):
        if _is_thematic_file(path):
            # skip broad topic pages
            continue

        # treat non-thematic files as flower-specific pages
        raw_text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw_text:
            # skip empty files (duh)
            continue

        display_name = _slug_to_display_name(path.stem)
        doc = _new_flower_doc(display_name)
        title = _extract_title(raw_text, display_name)
        # save page title as another alias for matching
        doc["aliases"].add(_normalize(title))
        doc["article_passages"].extend(_split_passages(raw_text))
        docs_by_key[doc["key"]] = doc

    for record in _load_csv_records():
        # merge only when we can match the CSV record to a known flower page
        key = _best_matching_key(record["name"], record["scientific_name"], docs_by_key)
        if key is not None:
            _merge_structured_record(docs_by_key[key], record)

    flowers = []
    for flower in sorted(docs_by_key.values(), key=lambda item: item["name"].lower()):
        # build and store the final searchable document for each flower
        passages, document = _build_document(flower)
        flower["passages"] = passages
        flower["document"] = document
        flowers.append(flower)
    # sorting keeps the model build consistent across runs
    return flowers


@lru_cache(maxsize=1)
def _load_model() -> tuple:
    # build the vectorizers and SVD model once, then reuse them for all queries (bc SVD expensive af)
    flowers = _build_corpus_docs()
    documents = [flower["document"] for flower in flowers]

    word_vectorizer = TfidfVectorizer(
        # word n-grams capture explicit words and short phrases
        preprocessor=_normalize,
        stop_words="english",
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        dtype=np.float32,
    )
    char_vectorizer = TfidfVectorizer(
        # character n-grams make the model less sensitive to spelling differences
        preprocessor=_normalize,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
        dtype=np.float32,
    )

    # first create sparse TF-IDF features, then compress them with SVD
    word_matrix = word_vectorizer.fit_transform(documents)
    char_matrix = char_vectorizer.fit_transform(documents)
    combined_matrix = hstack([word_matrix, char_matrix], format="csr")
    svd, lsa_matrix = _fit_lsa(combined_matrix, MAX_SVD_COMPONENTS)

    return (
        flowers,
        word_vectorizer,
        char_vectorizer,
        # keep the word-only matrix for explanation features shown in the UI
        # would be kinda weird if characters started showing up instead
        word_matrix,
        int(combined_matrix.shape[1]),
        svd,
        lsa_matrix,
    )


def model_info() -> dict:
    """
    Return basic information about the built model for debugging or inspection.
    """
    flowers, word_vectorizer, char_vectorizer, _, feature_count, svd, _ = _load_model()
    return {
        "retrieval_mode": "svd_only",
        "corpus_dir": str(TEXT_CORPUS_DIR),
        "document_count": len(flowers),
        "feature_count": feature_count,
        # these counts help confirm that the vectorizers were built correctly (i can't anymore this was 
        # supposed to just be for debugging but now that it's there it's working i got no clue why
        # so i'm not touching it)
        "word_feature_count": len(word_vectorizer.get_feature_names_out()),
        "char_feature_count": len(char_vectorizer.get_feature_names_out()),
        "svd_components": 0 if svd is None else int(svd.n_components),
    }

# FINALLY, HOLY MOOOLLLYY, turn the ranked flower docs into a frontend response format
def _build_suggestion(
    flower: dict,
    query: str,
    similarity: float,
    top_score: float,
    query_word_matrix,
    word_matrix,
    index: int,
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    svd: TruncatedSVD | None,
) -> dict:
    # convert one ranked flower into the response shape expected by the frontend
    matched_terms = _top_feature_terms(
        # multiply the query and flower word vectors to find shared explanation terms.
        query_word_matrix.multiply(word_matrix[index]),
        word_vectorizer,
        MAX_MATCHED_KEYWORDS,
    )
    for matched_term in matched_terms:
        matched_term["category"] = _keyword_category_for_flower(matched_term["keyword"], flower)
    displayed_meanings = _select_display_texts(flower["meanings"], query, 2)
    displayed_occasions = _select_display_texts(flower["occasions"], query, 2)

    if not displayed_meanings and not displayed_occasions:
        # if short metadata is missing, use passages as fallback explanation text
        highlights = _top_passages(query, flower["passages"], word_vectorizer, char_vectorizer, svd)
        displayed_meanings = highlights[:2]
        displayed_occasions = highlights[2:]

    return {
        "name": flower["name"],
        "scientific_name": flower["scientific_name"],
        "colors": flower["colors"],
        "plant_types": flower["plant_types"],
        "maintenance": flower["maintenance"],
        "meanings": displayed_meanings,
        "occasions": displayed_occasions,
        # this score is relative to the best result for the same query
        "score": round((similarity / top_score) * 100, 2),
        "matched_keywords": matched_terms,
    }

# NOW FOR THE ACTUAL THING HOLY
def recommend_flowers(query: str, limit: int = 5) -> dict:
    """
    Main public function:
     1. vectorize the query
     2. compare it to all flower documents in SVD space
     3. return the best matches in frontend format
    """
    if not query or not query.strip():
        return _empty_response(query)

    flowers, word_vectorizer, char_vectorizer, word_matrix, _, svd, lsa_matrix = _load_model()
    query_word_matrix = word_vectorizer.transform([query])
    query_combined_matrix = _combined_features([query], word_vectorizer, char_vectorizer)

    # if the query does not produce usable features, we cannot rank anything (no duh)
    if query_combined_matrix.nnz == 0 or svd is None or lsa_matrix is None:
        return _empty_response(query)

    # project the query into the same latent space as the flower documents,
    # then compare with cosine similarity
    query_lsa = normalize(svd.transform(query_combined_matrix), norm="l2")
    similarities = cosine_similarity(query_lsa, lsa_matrix)[0]
    ranked_indices = np.argsort(similarities)[::-1]

    # keep only the positive similarities. non-positive values can go bye bye
    positive_scores = [float(similarities[index]) for index in ranked_indices if float(similarities[index]) > 0]
    if not positive_scores:
        # even if no flower matches, we can still expose important query terms
        return _empty_response(
            query,
            _categorize_feature_terms_for_corpus(
                _top_feature_terms(query_word_matrix, word_vectorizer, MAX_QUERY_KEYWORDS),
                flowers,
            ),
        )

    top_score = positive_scores[0]
    suggestions = []
    for index in ranked_indices:
        similarity = float(similarities[index])
        if len(suggestions) >= limit or similarity <= 0:
            break
        # normalize each shown score relative to the top result for this query.
        suggestions.append(
            _build_suggestion(
                flowers[index],
                query,
                similarity,
                top_score,
                query_word_matrix,
                word_matrix,
                index,
                word_vectorizer,
                char_vectorizer,
                svd,
            )
        )

    # FINAALLLLYYYYY, WE CAN RETURN THE SUGGESTIONS OHHHH MY LAWWWWD THERE HAS TO BE AN EASIER WAY TO DO THIS
    return {
        "query": query,
        "keywords_used": _categorize_feature_terms_for_corpus(
            _top_feature_terms(query_word_matrix, word_vectorizer, MAX_QUERY_KEYWORDS),
            flowers,
        ),
        "suggestions": suggestions,
    }
