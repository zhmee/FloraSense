"""
FloraSense prototype 3: pure latent retrieval with TF-IDF + SVD.

This file builds one text document for each flower, turns those documents into
numeric vectors, reduces them with SVD, and then finds the flowers whose vectors
are closest to the user's query.

`merged.csv` is the primary source of flower records. Scraped flower pages are
used only to add article text when they can be matched to a cleaned CSV flower.
"""

from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from flower_radar_chart import build_latent_radar_chart, select_latent_axes


# the file locations
DATA_FILE = Path(__file__).resolve().parent / "data" / "merged.csv"
TEXT_CORPUS_DIR = Path(__file__).resolve().parent.parent / "data_scraping" / "flower_texts"
FLOWER_IMAGE_DIR = Path(__file__).resolve().parent.parent / "data_scraping" / "flower_images"

# optional external map of scraped page keys -> image URLs
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
_COLOR_FALLBACK_IMAGE_FILENAMES = {
    "blue": "blue-flowers-meaning.jpg",
    "pink": "pink-flowers-meaning.jpg",
    "purple": "purple-flowers-meaning.jpg",
    "white": "white-flowers.jpg",
    "yellow": "yellow-flowers-meaning.jpg",
}
_GENERIC_FLOWER_IMAGE_FILENAME = "10-most-beautiful-flowers.jpg"
ENABLE_QUERY_LATENT_DEBUG = False

# configuration values for the model and for the returned UI payload.
MAX_SVD_COMPONENTS = 96
MAX_QUERY_KEYWORDS = 10
MAX_MATCHED_KEYWORDS = 8
MAX_HIGHLIGHTS = 3
QUERY_AXIS_FALLBACK_ATTENUATION = 0.72
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


def _normalize_image_key(text: str) -> str:
    """
    Normalize image slugs with a small typo fix for scraped filenames.
    """
    return _normalize(text).replace("lilly", "lily")


def _is_thematic_slug(slug: str) -> bool:
    return slug.endswith("flowers") or slug.startswith(THEMATIC_PREFIXES)


def _load_local_image_index(include_thematic: bool = True) -> tuple[dict[str, str], set[str]]:
    try:
        image_map: dict[str, str] = {}
        filenames: set[str] = set()
        for path in FLOWER_IMAGE_DIR.iterdir():
            if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            if not include_thematic and _is_thematic_slug(path.stem):
                continue
            normalized_key = _normalize_image_key(path.stem)
            if normalized_key:
                image_map[normalized_key] = path.name
            filenames.add(path.name)
        return image_map, filenames
    except Exception:
        return {}, set()


_LOCAL_IMAGE_FILES, _LOCAL_IMAGE_FILENAMES = _load_local_image_index(include_thematic=True)
_LOCAL_FLOWER_IMAGE_FILES, _ = _load_local_image_index(include_thematic=False)


def _tokenize(text: str) -> list[str]:
    """
    Split normalized text into basic word-like tokens.
    """
    return TOKEN_PATTERN.findall(_normalize(text))


def _token_root_variants(token: str) -> set[str]:
    """
    Build a small family of normalized token variants so explanation labels can
    recover readable roots like "love" from inflected forms like "loveliness".
    """
    normalized_token = _normalize(token)
    if not normalized_token or " " in normalized_token:
        return set()

    variants: set[str] = set()
    pending = [normalized_token]

    def add_variant(value: str) -> None:
        candidate = _normalize(value)
        if not candidate or " " in candidate or len(candidate) < 3 or candidate in variants:
            return
        pending.append(candidate)

    while pending:
        current = pending.pop()
        if current in variants:
            continue
        variants.add(current)

        if len(current) <= 3:
            continue

        if current.endswith("iness") and len(current) > 6:
            base = current[:-5]
            add_variant(base)
            add_variant(base + "y")
            if base.endswith("l"):
                add_variant(base[:-1] + "e")

        if current.endswith("ness") and len(current) > 6:
            base = current[:-4]
            add_variant(base)
            if base.endswith("i"):
                add_variant(base[:-1] + "y")
            if base.endswith("l"):
                add_variant(base[:-1] + "e")

        if current.endswith("ly") and len(current) > 4:
            base = current[:-2]
            add_variant(base)
            if base.endswith("i"):
                add_variant(base[:-1] + "y")
            if base.endswith("l"):
                add_variant(base[:-1] + "e")

        if current.endswith("ing") and len(current) > 5:
            base = current[:-3]
            add_variant(base)
            add_variant(base + "e")
            if len(base) >= 2 and base[-1] == base[-2]:
                add_variant(base[:-1])

        if current.endswith("ed") and len(current) > 4:
            base = current[:-2]
            add_variant(base)
            add_variant(base + "e")
            if len(base) >= 2 and base[-1] == base[-2]:
                add_variant(base[:-1])

        if current.endswith("ies") and len(current) > 4:
            add_variant(current[:-3] + "y")

        if current.endswith("es") and len(current) > 4:
            add_variant(current[:-2])
            add_variant(current[:-1])

        if current.endswith("s") and len(current) > 3 and not current.endswith("ss"):
            add_variant(current[:-1])

    return variants


def _tokens_share_root(left: str, right: str) -> bool:
    left_variants = _token_root_variants(left)
    right_variants = _token_root_variants(right)
    if not left_variants or not right_variants:
        return False
    return bool(left_variants & right_variants)


def _phrase_match_count(left_tokens: list[str], right_tokens: list[str]) -> int:
    """
    Count token-level matches between two phrases, allowing simple
    morphology-aware matches on each token.
    """
    match_count = 0
    used_right_indices = set()

    for left_token in left_tokens:
        for index, right_token in enumerate(right_tokens):
            if index in used_right_indices:
                continue
            if _tokens_share_root(left_token, right_token):
                used_right_indices.add(index)
                match_count += 1
                break

    return match_count


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


def _text_similarity(left: str, right: str) -> float:
    left_normalized = _normalize(left)
    right_normalized = _normalize(right)
    if not left_normalized or not right_normalized:
        return 0.0
    return float(SequenceMatcher(None, left_normalized, right_normalized).ratio())


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


def _matching_image_candidates(flower: dict) -> list[tuple[str, str]]:
    candidates = []
    flower_key = _normalize_image_key(flower.get("name", ""))
    if flower_key:
        candidates.append(("name", flower_key))
    for alias in flower.get("aliases", set()):
        normalized_alias = _normalize_image_key(alias)
        if normalized_alias:
            candidates.append(("alias", normalized_alias))
    return candidates


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


def _fallback_image_url_from_metadata(flower: dict) -> str | None:
    for color in flower.get("colors", []):
        normalized_color = _normalize(color)
        filename = _COLOR_FALLBACK_IMAGE_FILENAMES.get(normalized_color)
        if filename in _LOCAL_IMAGE_FILENAMES:
            return f"/api/flower-images/{filename}"

    if _GENERIC_FLOWER_IMAGE_FILENAME in _LOCAL_IMAGE_FILENAMES:
        return f"/api/flower-images/{_GENERIC_FLOWER_IMAGE_FILENAME}"

    return None


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
        "query_latent_radar_chart": None,
        "query_latent_radar_axes": [],
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


def _load_scraped_docs() -> dict[str, dict]:
    """
    Load one base flower document per non-thematic scraped text file.
    """
    docs_by_key: dict[str, dict] = {}

    for path in sorted(TEXT_CORPUS_DIR.glob("*.txt")):
        if _is_thematic_file(path):
            # skip broad topic pages
            continue

        raw_text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw_text:
            # skip empty files
            continue

        display_name = _slug_to_display_name(path.stem)
        doc = _new_flower_doc(display_name)
        title = _extract_title(raw_text, display_name)
        # keep the page title as another possible name for matching
        doc["aliases"].add(_normalize(title))
        doc["article_passages"].extend(_split_passages(raw_text))
        docs_by_key[doc["key"]] = doc

    return docs_by_key


# BASICALLY, ALL THIS IS JUST TO MATCH THE DIFF DATA SOURCES TO THE SAME FLOWER TO BUILD THE FLOWER DOCUMENTS
def _best_matching_key(
    name: str,
    scientific_name: str,
    docs_by_key: dict[str, dict],
    extra_candidates: set[str] | None = None,
) -> str | None:
    """
    Find which existing flower document best matches a CSV row.
    This is needed because the article files and the CSV are not always named
    in exactly the same way (which really pmo man I was wondering what was going on for so long).
    """
    candidate_texts = [text for text in (_normalize(name), _normalize(scientific_name)) if text]
    for candidate in extra_candidates or set():
        normalized_candidate = _normalize(candidate)
        if normalized_candidate and normalized_candidate not in candidate_texts:
            candidate_texts.append(normalized_candidate)
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


def _merge_scraped_record(doc: dict, scraped_doc: dict) -> None:
    """
    Add scraped article text into an existing flower document when available.
    """
    doc["aliases"].update(scraped_doc["aliases"])
    doc["article_passages"] = _dedupe_preserve_order(
        doc["article_passages"] + scraped_doc["article_passages"]
    )


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
    # if there;s actual metadata, we want that clean text
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
    # TRUNCATEDSVD IS THE GOAT TRUST we use it to reduce the high-dimensional TF-IDF matrix
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


def _merge_keyword_lists(term_groups: list[list[dict]], limit: int) -> list[dict]:
    """
    Merge multiple keyword candidate lists while keeping the first clear,
    non-redundant terms.
    """
    merged = []

    for group in term_groups:
        for term in group:
            keyword = term["keyword"]
            if _is_generic_flower_term(keyword):
                continue

            keyword_tokens = set(_tokenize(keyword))
            keyword_category = term.get("category")
            if not keyword_tokens:
                continue

            replacement_indices = []
            skip_term = False
            for index, existing_term in enumerate(merged):
                if existing_term.get("category") != keyword_category:
                    continue

                existing_keyword = existing_term["keyword"]
                existing_tokens = set(_tokenize(existing_keyword))
                if not existing_tokens:
                    continue

                if _normalize(existing_keyword) == _normalize(keyword) or keyword_tokens <= existing_tokens:
                    skip_term = True
                    break

                if existing_tokens < keyword_tokens:
                    replacement_indices.append(index)

            if skip_term:
                continue

            selected_keywords = [
                existing_term["keyword"]
                for index, existing_term in enumerate(merged)
                if index not in replacement_indices
            ]
            if _is_redundant_keyword(keyword, selected_keywords):
                continue

            insert_at = len(merged) if not replacement_indices else replacement_indices[0]
            for index in reversed(replacement_indices):
                merged.pop(index)
            merged.insert(insert_at, term)
            if len(merged) >= limit:
                return merged

    return merged


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
            if _phrase_match_count(list(keyword_tokens), list(value_tokens)) == min(len(keyword_tokens), len(value_tokens)):
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


def _metadata_query_phrase_variants(category: str, value: str) -> list[tuple[str, str]]:
    """
    Return query phrases to look for plus the display label to expose in the UI.
    """
    normalized_value = _normalize(value)
    if not normalized_value:
        return []

    variants = [(normalized_value, normalized_value)]
    if category == "maintenance" and not normalized_value.endswith("maintenance"):
        variants.insert(0, (f"{normalized_value} maintenance", f"{normalized_value} maintenance"))

    seen = set()
    deduped = []
    for phrase, display in variants:
        key = (phrase, display)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((phrase, display))
    return deduped


def _extract_query_metadata_keywords(query: str, flowers: list[dict]) -> list[dict]:
    """
    Pull explicit metadata phrases directly out of the raw query so the
    query breakdown can show exact attributes like "pink" and
    "low maintenance" instead of only composite TF-IDF phrases.
    """
    normalized_query = _normalize(query)
    if not normalized_query:
        return []
    query_tokens = normalized_query.split()
    token_spans = [(match.group(0), match.start(), match.end()) for match in re.finditer(r"[a-z0-9]+", normalized_query)]

    category_values: dict[str, set[str]] = {
        "color": set(),
        "maintenance": set(),
        "plant_type": set(),
    }
    for flower in flowers:
        category_values["color"].update(_normalize(value) for value in flower["colors"] if _normalize(value))
        category_values["maintenance"].update(_normalize(value) for value in flower["maintenance"] if _normalize(value))
        category_values["plant_type"].update(_normalize(value) for value in flower["plant_types"] if _normalize(value))

    candidates = []
    for category, values in category_values.items():
        for value in sorted(values):
            for phrase, display in _metadata_query_phrase_variants(category, value):
                pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)")
                for match in pattern.finditer(normalized_query):
                    candidates.append(
                        {
                            "keyword": display,
                            "category": category,
                            "score": round(10 + len(phrase.split()) / 10, 2),
                            "start": match.start(),
                            "end": match.end(),
                            "is_exact": True,
                        }
                    )

                phrase_tokens = phrase.split()
                if not phrase_tokens or len(query_tokens) < len(phrase_tokens):
                    continue

                best_fuzzy_match = None
                best_similarity = 0.0
                for start_index in range(0, len(query_tokens) - len(phrase_tokens) + 1):
                    end_index = start_index + len(phrase_tokens)
                    window_text = " ".join(query_tokens[start_index:end_index])
                    similarity = _text_similarity(window_text, phrase)
                    if similarity <= best_similarity:
                        continue
                    best_similarity = similarity
                    best_fuzzy_match = (start_index, end_index)

                min_similarity = 0.84 if len(phrase_tokens) == 1 else 0.8
                if best_fuzzy_match is None or best_similarity < min_similarity or best_similarity >= 0.999:
                    continue

                start_token_index, end_token_index = best_fuzzy_match
                fuzzy_start = token_spans[start_token_index][1]
                fuzzy_end = token_spans[end_token_index - 1][2]
                candidates.append(
                    {
                        "keyword": display,
                        "category": category,
                        "score": round(9.4 + best_similarity + len(phrase_tokens) / 10, 2),
                        "start": fuzzy_start,
                        "end": fuzzy_end,
                        "is_exact": False,
                    }
                )

    selected = []
    occupied_ranges: list[tuple[int, int]] = []
    seen_keywords = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -(item["end"] - item["start"]),
            -int(item.get("is_exact", False)),
            -item["score"],
            item["start"],
            item["keyword"],
        ),
    ):
        overlaps_existing = any(
            candidate["start"] < existing_end and candidate["end"] > existing_start
            for existing_start, existing_end in occupied_ranges
        )
        if overlaps_existing:
            continue

        keyword_key = (candidate["category"], _normalize(candidate["keyword"]))
        if keyword_key in seen_keywords:
            continue

        seen_keywords.add(keyword_key)
        occupied_ranges.append((candidate["start"], candidate["end"]))
        selected.append(candidate)

    selected.sort(key=lambda item: (item["start"], item["category"], item["keyword"]))
    return [
        {
            "keyword": item["keyword"],
            "category": item["category"],
            "score": item["score"],
        }
        for item in selected
    ]


def _collect_keyword_candidates(flowers: list[dict] | None = None, flower: dict | None = None) -> list[tuple[str, str]]:
    """
    Collect short, human-readable candidate phrases from flower metadata for
    query explanation matching.
    """
    candidate_pairs = []
    seen_pairs = set()

    flower_iterable = [flower] if flower is not None else flowers or []
    for item in flower_iterable:
        raw_values = {
            "color": item.get("colors", []),
            "maintenance": item.get("maintenance", []),
            "plant_type": item.get("plant_types", []),
            "occasion": item.get("occasions", []),
            "meaning": item.get("meanings", []),
        }

        for category, values in raw_values.items():
            for value in values:
                if category in {"meaning", "occasion"}:
                    chunks = _split_meaning_chunks(value)
                else:
                    chunks = [value]

                for chunk in chunks:
                    normalized_chunk = _normalize(chunk)
                    chunk_tokens = _tokenize(normalized_chunk)
                    if (
                        not normalized_chunk
                        or not chunk_tokens
                        or len(chunk_tokens) > 3
                        or _is_generic_flower_term(normalized_chunk)
                    ):
                        continue

                    pair = (category, normalized_chunk)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    candidate_pairs.append(pair)

    return candidate_pairs


def _extract_query_morphology_keywords(
    query: str,
    candidate_pairs: list[tuple[str, str]],
    limit: int,
) -> list[dict]:
    """
    Recover readable explanation terms from raw query morphology, so forms like
    "loveliness" can still surface as "love" in the UI.
    """
    query_tokens = _tokenize(query)
    normalized_query = _normalize(query)
    if not normalized_query or not query_tokens:
        return []

    category_priority = {
        "color": 0,
        "maintenance": 1,
        "plant_type": 2,
        "occasion": 3,
        "meaning": 4,
    }

    candidates = []
    for category, label in candidate_pairs:
        label_tokens = _tokenize(label)
        if not label_tokens:
            continue

        match_count = _phrase_match_count(query_tokens, label_tokens)
        if match_count < len(label_tokens):
            continue

        exact_phrase_match = bool(re.search(rf"(?<!\w){re.escape(label)}(?!\w)", normalized_query))
        candidates.append(
            {
                "keyword": label,
                "category": category,
                "score": round(8.5 + (0.4 if exact_phrase_match else 0.0) + len(label_tokens) / 10, 2),
                "exact": exact_phrase_match,
                "priority": category_priority.get(category, 99),
                "token_count": len(label_tokens),
            }
        )

    selected = []
    seen_keywords = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (-int(item["exact"]), -item["token_count"], item["priority"], item["keyword"]),
    ):
        keyword_key = (candidate["category"], candidate["keyword"])
        if keyword_key in seen_keywords:
            continue
        seen_keywords.add(keyword_key)
        selected.append(
            {
                "keyword": candidate["keyword"],
                "category": candidate["category"],
                "score": candidate["score"],
            }
        )
        if len(selected) >= limit:
            break

    return selected


def _build_query_breakdown_keywords(
    query: str,
    query_word_matrix,
    word_vectorizer: TfidfVectorizer,
    flowers: list[dict],
    limit: int,
) -> list[dict]:
    """
    Combine exact metadata phrases from the query with the strongest semantic terms.
    """
    metadata_keywords = _extract_query_metadata_keywords(query, flowers)
    morphology_keywords = _extract_query_morphology_keywords(
        query,
        _collect_keyword_candidates(flowers=flowers),
        limit,
    )
    semantic_keywords = _categorize_feature_terms_for_corpus(
        _top_feature_terms(query_word_matrix, word_vectorizer, limit),
        flowers,
    )

    return _merge_keyword_lists(
        [metadata_keywords, morphology_keywords, semantic_keywords],
        limit,
    )


def _axis_label_key(label: str) -> str:
    return _normalize(" ".join((label or "").replace("\n", " ").split()))


def _is_maintenance_axis_label(label: str) -> bool:
    return "maintenance" in set(_tokenize(label))


def _build_query_axis_candidates(
    query_keywords: list[dict],
    word_vectorizer: TfidfVectorizer,
) -> list[dict]:
    vocabulary = word_vectorizer.vocabulary_
    query_candidates = []
    seen_candidates = set()

    for item in query_keywords:
        label = item["keyword"].strip()
        normalized_label = _normalize(label)
        tokens = _tokenize(label)
        if not normalized_label or normalized_label in seen_candidates or not tokens:
            continue

        feature_indices = []
        for size in range(min(3, len(tokens)), 0, -1):
            for start in range(0, len(tokens) - size + 1):
                phrase = " ".join(tokens[start : start + size])
                feature_index = vocabulary.get(phrase)
                if feature_index is not None:
                    feature_indices.append(int(feature_index))

        if not feature_indices:
            continue

        seen_candidates.add(normalized_label)
        query_candidates.append(
            {
                "label": label,
                "normalized": normalized_label,
                "category": item.get("category", "semantic"),
                "feature_indices": sorted(set(feature_indices)),
            }
        )

    return query_candidates


def _select_query_latent_axes(
    vector: np.ndarray,
    component_labels: list[str],
    query_keywords: list[dict],
    svd: TruncatedSVD | None,
    word_vectorizer: TfidfVectorizer,
    axis_limit: int = 6,
) -> dict | None:
    """
    Prefer latent components that best support explicit query attributes, then
    fill the remaining slots with the normal strongest components.
    """
    profile_array = np.asarray(vector, dtype=np.float32).ravel()
    fallback_axes = select_latent_axes(profile_array, component_labels, axis_limit)
    if fallback_axes is None:
        return None

    if svd is None or not query_keywords:
        return fallback_axes

    word_feature_names = word_vectorizer.get_feature_names_out()
    if len(word_feature_names) == 0:
        return fallback_axes

    query_candidates = _build_query_axis_candidates(query_keywords, word_vectorizer)
    if not query_candidates:
        return fallback_axes

    ranked_indices = np.argsort(np.abs(profile_array))[::-1]
    selected_indices: list[int] = []
    selected_labels: list[str] = []
    used_indices = set()
    used_label_keys = set()
    selected_candidate_categories = set()

    for candidate in query_candidates:
        best_index = None
        best_score = (-1.0, -1.0, -1.0)

        for axis_index in ranked_indices:
            axis_index = int(axis_index)
            if axis_index in used_indices or axis_index >= svd.components_.shape[0]:
                continue

            word_loadings = svd.components_[axis_index, : len(word_feature_names)]
            top_loading = float(np.max(np.abs(word_loadings)))
            if top_loading <= 0:
                continue

            candidate_loading = max(
                float(np.abs(word_loadings[feature_index]))
                for feature_index in candidate["feature_indices"]
            )
            loading_ratio = candidate_loading / top_loading
            if loading_ratio < 0.18:
                continue

            category_bonus = 1.0 if candidate["category"] != "semantic" else 0.0
            candidate_score = (
                loading_ratio,
                category_bonus,
                float(np.abs(profile_array[axis_index])),
            )
            if candidate_score > best_score:
                best_score = candidate_score
                best_index = axis_index

        label_key = _axis_label_key(candidate["label"])
        if best_index is None or not label_key or label_key in used_label_keys:
            continue

        selected_indices.append(best_index)
        selected_labels.append(candidate["label"])
        used_indices.add(best_index)
        used_label_keys.add(label_key)
        selected_candidate_categories.add(candidate["category"])
        if len(selected_indices) >= min(axis_limit, profile_array.size):
            break

    for axis_index, axis_label in zip(
        np.asarray(fallback_axes["axis_indices"], dtype=int),
        fallback_axes["axis_labels"],
    ):
        axis_index = int(axis_index)
        label_key = _axis_label_key(axis_label)
        if axis_index in used_indices or label_key in used_label_keys:
            continue
        if (
            "maintenance" in selected_candidate_categories
            and _is_maintenance_axis_label(axis_label)
        ):
            continue
        selected_indices.append(axis_index)
        selected_labels.append(axis_label)
        used_indices.add(axis_index)
        used_label_keys.add(label_key)
        if len(selected_indices) >= min(axis_limit, profile_array.size):
            break

    if len(selected_indices) < 3:
        return fallback_axes

    return {
        "axis_indices": np.asarray(selected_indices, dtype=int),
        "axis_labels": selected_labels,
    }


def _query_axis_display_values(
    vector: np.ndarray,
    axis_indices: np.ndarray,
    axis_labels: list[str],
    query_keywords: list[dict],
) -> np.ndarray:
    """
    Make explicitly queried axes visually primary on the query radar while
    keeping the underlying latent axis selection unchanged.
    """
    profile_array = np.asarray(vector, dtype=np.float32).ravel()
    axis_indices = np.asarray(axis_indices, dtype=int)
    display_values = np.abs(profile_array[axis_indices]).astype(np.float32, copy=True)
    if display_values.size == 0:
        return display_values

    queried_label_keys = {
        _axis_label_key(item["keyword"])
        for item in query_keywords
        if _axis_label_key(item.get("keyword", ""))
    }
    if not queried_label_keys:
        return display_values

    queried_positions = [
        position
        for position, label in enumerate(axis_labels)
        if _axis_label_key(label) in queried_label_keys
    ]
    if not queried_positions:
        return display_values

    target_value = float(max(np.max(display_values), 1e-6))
    for position in queried_positions:
        display_values[position] = max(float(display_values[position]), target_value)
    for position in range(display_values.size):
        if position in queried_positions:
            continue
        display_values[position] = min(
            float(display_values[position]),
            target_value * QUERY_AXIS_FALLBACK_ATTENUATION,
        )

    return display_values


def _flower_axis_display_values(
    vector: np.ndarray,
    axis_indices: np.ndarray,
    axis_labels: list[str],
    matched_terms: list[dict],
) -> np.ndarray:
    """
    Make flower-card radar charts emphasize the query-matched axes that the
    flower actually supports, while leaving unmatched fallback axes secondary.
    """
    profile_array = np.asarray(vector, dtype=np.float32).ravel()
    axis_indices = np.asarray(axis_indices, dtype=int)
    display_values = np.abs(profile_array[axis_indices]).astype(np.float32, copy=True)
    if display_values.size == 0:
        return display_values

    matched_label_keys = {
        _axis_label_key(item["keyword"])
        for item in matched_terms
        if _axis_label_key(item.get("keyword", ""))
    }
    if not matched_label_keys:
        return display_values

    matched_positions = [
        position
        for position, label in enumerate(axis_labels)
        if _axis_label_key(label) in matched_label_keys
    ]
    if not matched_positions:
        return display_values

    target_value = float(max(np.max(display_values), 1e-6))
    for position in matched_positions:
        display_values[position] = max(float(display_values[position]), target_value)
    for position in range(display_values.size):
        if position in matched_positions:
            continue
        display_values[position] = min(
            float(display_values[position]),
            target_value * QUERY_AXIS_FALLBACK_ATTENUATION,
        )

    return display_values


def _relabel_query_axes(
    axis_indices: np.ndarray,
    fallback_axis_labels: list[str],
    query_keywords: list[dict],
    svd: TruncatedSVD | None,
    word_vectorizer: TfidfVectorizer,
) -> list[str]:
    """
    Prefer exact query-matched labels for selected latent axes when those
    query terms have meaningful word-feature loadings on the same components.
    """
    if svd is None or len(axis_indices) == 0 or not query_keywords:
        return fallback_axis_labels

    word_feature_names = word_vectorizer.get_feature_names_out()
    if len(word_feature_names) == 0:
        return fallback_axis_labels

    query_candidates = _build_query_axis_candidates(query_keywords, word_vectorizer)
    if not query_candidates:
        return fallback_axis_labels

    relabeled_axes = list(fallback_axis_labels)
    used_candidate_labels = set()

    for axis_position, axis_index in enumerate(np.asarray(axis_indices, dtype=int)):
        if axis_index >= svd.components_.shape[0]:
            continue

        word_loadings = svd.components_[int(axis_index), : len(word_feature_names)]
        top_loading = float(np.max(np.abs(word_loadings)))
        if top_loading <= 0:
            continue

        occupied_label_keys = {
            _axis_label_key(label)
            for label_position, label in enumerate(relabeled_axes)
            if label_position != axis_position
        }
        best_candidate = None
        best_score = (-1.0, -1.0, -1.0)
        for candidate in query_candidates:
            candidate_label_key = _axis_label_key(candidate["label"])
            if (
                candidate["normalized"] in used_candidate_labels
                or not candidate_label_key
                or candidate_label_key in occupied_label_keys
            ):
                continue

            candidate_loading = max(
                float(np.abs(word_loadings[feature_index]))
                for feature_index in candidate["feature_indices"]
            )
            loading_ratio = candidate_loading / top_loading
            if loading_ratio < 0.18:
                continue

            category_bonus = 1.0 if candidate["category"] != "semantic" else 0.0
            candidate_score = (
                loading_ratio,
                category_bonus,
                float(len(candidate["normalized"])),
            )
            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = candidate

        if best_candidate is None:
            continue

        used_candidate_labels.add(best_candidate["normalized"])
        relabeled_axes[axis_position] = best_candidate["label"]

    return relabeled_axes


def _print_query_latent_dimensions(
    query: str,
    query_vector: np.ndarray,
    axis_indices: np.ndarray,
    axis_labels: list[str],
) -> None:
    """
    Debug-only terminal print for the selected latent dimensions behind the query radar.
    """
    if len(axis_indices) == 0 or not axis_labels:
        return

    entries = []
    profile_array = np.asarray(query_vector, dtype=np.float32).ravel()
    for axis_index, axis_label in zip(np.asarray(axis_indices, dtype=int), axis_labels):
        if int(axis_index) >= profile_array.size:
            continue
        axis_value = float(profile_array[int(axis_index)])
        entries.append(f"{int(axis_index)}:{axis_label}={axis_value:.4f}")

    if entries:
        print(f"[latent dims] query={query!r} -> " + ", ".join(entries))


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
    return _is_thematic_slug(path.stem)


def _build_corpus_docs() -> list[dict]:
    """
    Build the flower documents from cleaned CSV records, then attach scraped
    article text when a flower page can be matched back to that CSV flower.
    """
    docs_by_key: dict[str, dict] = {}

    for record in _load_csv_records():
        doc = _new_flower_doc(record["name"], record["scientific_name"])
        _merge_structured_record(doc, record)
        docs_by_key[doc["key"]] = doc

    scraped_docs_by_key = _load_scraped_docs()
    for scraped_doc in scraped_docs_by_key.values():
        key = _best_matching_key(
            scraped_doc["name"],
            scraped_doc["scientific_name"],
            docs_by_key,
            extra_candidates=scraped_doc["aliases"],
        )
        if key is None:
            continue
        _merge_scraped_record(docs_by_key[key], scraped_doc)

    flowers = []
    for flower in sorted(docs_by_key.values(), key=lambda item: item["name"].lower()):
        # build and store the final searchable document for each flower
        passages, document = _build_document(flower)
        flower["passages"] = passages
        flower["document"] = document
        flowers.append(flower)
    # sorting keeps the model build consistent across runs

    # attach image_url to each flower using the optional image map (tolerant matching)
    for flower in flowers:
        flower["image_url"] = _resolve_flower_image_url(flower)

    return flowers


def _component_labels_from_svd(
    svd: TruncatedSVD | None,
    word_vectorizer: TfidfVectorizer,
    flowers: list[dict],
    max_terms: int = 3,
) -> list[str]:
    """
    Give each latent dimension a human-readable label based on its
    strongest word features.
    """
    if svd is None:
        return []

    word_feature_names = word_vectorizer.get_feature_names_out()
    if len(word_feature_names) == 0:
        return []

    name_like_terms = set()
    descriptor_candidates = []
    seen_descriptors = set()

    def add_descriptor(value: str) -> None:
        normalized_value = _normalize(value)
        tokens = set(_tokenize(value))
        if (
            not normalized_value
            or normalized_value in seen_descriptors
            or not tokens
            or normalized_value in name_like_terms
            or _is_generic_flower_term(normalized_value)
        ):
            return
        seen_descriptors.add(normalized_value)
        descriptor_candidates.append(
            {
                "label": value.strip(),
                "normalized": normalized_value,
                "tokens": tokens,
            }
        )

    for flower in flowers:
        alias_values = {
            flower["name"],
            flower["scientific_name"],
            *flower["aliases"],
        }
        for alias in alias_values:
            normalized_alias = _normalize(alias)
            if normalized_alias:
                name_like_terms.add(normalized_alias)
            for token in _tokenize(alias):
                if len(token) >= 4:
                    name_like_terms.add(token)

    for flower in flowers:
        for maintenance in flower["maintenance"]:
            add_descriptor(maintenance)
            add_descriptor(f"{maintenance} maintenance")
        for color in flower["colors"]:
            add_descriptor(color)
        for plant_type in flower["plant_types"]:
            add_descriptor(plant_type)
        for meaning in flower["meanings"]:
            for chunk in _split_meaning_chunks(meaning):
                if 1 <= len(chunk.split()) <= max_terms:
                    add_descriptor(chunk)
        for occasion in flower["occasions"]:
            for chunk in _split_meaning_chunks(occasion):
                if 1 <= len(chunk.split()) <= max_terms:
                    add_descriptor(chunk)

    labels = []
    for component in svd.components_:
        word_loadings = component[: len(word_feature_names)]
        ranked_indices = np.argsort(np.abs(word_loadings))[::-1]
        component_terms = [
            word_feature_names[int(index)]
            for index in ranked_indices[:40]
        ]
        component_term_ranks = {}
        for rank, term in enumerate(component_terms):
            normalized_component_term = _normalize(term)
            if not normalized_component_term:
                continue
            component_term_ranks.setdefault(normalized_component_term, rank)
        component_term_tokens = [
            set(_tokenize(term))
            for term in component_terms
        ]
        label = None

        for index in ranked_indices[:40]:
            term = word_feature_names[int(index)]
            normalized_term = _normalize(term)
            if (
                not normalized_term
                or _is_generic_flower_term(term)
                or normalized_term in name_like_terms
                or any(token in name_like_terms for token in _tokenize(term))
            ):
                continue

            term_tokens = set(_tokenize(term))
            best_descriptor = None
            best_descriptor_score = (-1, -1, -1, -1)
            for candidate in descriptor_candidates:
                candidate_tokens = candidate["tokens"]
                overlap = len(candidate_tokens & term_tokens)
                if overlap == 0:
                    continue
                if candidate_tokens <= term_tokens or term_tokens <= candidate_tokens:
                    exact_term_rank = component_term_ranks.get(candidate["normalized"])
                    component_support = max(
                        (
                            len(candidate_tokens & top_term_tokens)
                            for top_term_tokens in component_term_tokens
                        ),
                        default=0,
                    )
                    score = (
                        int(exact_term_rank is not None),
                        -999 if exact_term_rank is None else -exact_term_rank,
                        component_support,
                        overlap,
                        len(candidate["normalized"]),
                    )
                    if score > best_descriptor_score:
                        best_descriptor_score = score
                        best_descriptor = candidate["label"]

            if best_descriptor:
                label = best_descriptor
                break

            label = term
            break

        labels.append(label or "semantic signal")

    return labels


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
    component_labels = _component_labels_from_svd(svd, word_vectorizer, flowers)

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
        component_labels,
    )


def model_info() -> dict:
    """
    Return basic information about the built model for debugging or inspection.
    """
    flowers, word_vectorizer, char_vectorizer, _, feature_count, svd, _, _ = _load_model()
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


def _metadata_richness(flower: dict) -> tuple[int, int]:
    rich_count = sum(
        len(flower[field])
        for field in ("colors", "plant_types", "maintenance", "meanings", "occasions")
    )
    passage_count = len(flower.get("passages", []))
    return rich_count, passage_count


def _normalize_visualizer_positions(vectors: np.ndarray) -> np.ndarray:
    """
    Map latent vectors into a stable [-1, 1] cube for frontend layout.
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    dims = min(3, array.shape[1])
    trimmed = array[:, :dims]
    if dims < 3:
        trimmed = np.pad(trimmed, ((0, 0), (0, 3 - dims)), mode="constant")

    max_abs = np.max(np.abs(trimmed), axis=0)
    max_abs[max_abs < 1e-6] = 1.0
    return trimmed / max_abs


def visualizer_flowers(limit: int = 48) -> dict:
    """
    Return a lightweight flower graph dataset for the 3D visualizer.
    Uses the learned SVD latent vectors for base positions.
    """
    flowers, _, _, _, _, _, lsa_matrix, component_labels = _load_model()
    if lsa_matrix is None or len(flowers) == 0:
        return {"flowers": []}

    ranked_indices = sorted(
        range(len(flowers)),
        key=lambda index: (
            _metadata_richness(flowers[index])[0],
            _metadata_richness(flowers[index])[1],
            flowers[index]["name"].lower(),
        ),
        reverse=True,
    )
    chosen_indices = ranked_indices[: max(1, min(limit, len(ranked_indices)))]
    chosen_vectors = np.asarray([lsa_matrix[index] for index in chosen_indices], dtype=np.float32)
    normalized_positions = _normalize_visualizer_positions(chosen_vectors)

    dataset = []
    for output_index, flower_index in enumerate(chosen_indices):
        flower = flowers[flower_index]
        axis_info = select_latent_axes(lsa_matrix[flower_index], component_labels)
        primary_color = flower["colors"][0] if flower["colors"] else "neutral"
        primary_meaning = flower["meanings"][0] if flower["meanings"] else "general"
        primary_occasion = flower["occasions"][0] if flower["occasions"] else "everyday"
        x, y, z = normalized_positions[output_index]

        dataset.append(
            {
                "id": _normalize(flower["name"]).replace(" ", "-"),
                "name": flower["name"],
                "scientific_name": flower["scientific_name"],
                "colors": flower["colors"],
                "plant_types": flower["plant_types"],
                "maintenance": flower["maintenance"],
                "meanings": flower["meanings"],
                "occasions": flower["occasions"],
                "primary_color": primary_color,
                "primary_meaning": primary_meaning,
                "primary_occasion": primary_occasion,
                "latent_axes": [] if axis_info is None else axis_info["axis_labels"],
                "latent_position": {
                    "x": round(float(x), 4),
                    "y": round(float(y), 4),
                    "z": round(float(z), 4),
                },
                "image_url": flower.get("image_url"),
                "summary": _build_structured_passages(flower)[:3],
            }
        )

    return {"flowers": dataset}

# FINALLY, HOLY MOOOLLLYY, turn the ranked flower docs into a frontend response format
def _build_suggestion(
    flower: dict,
    query: str,
    similarity: float,
    top_score: float,
    query_lsa_vector: np.ndarray,
    flower_lsa_vector: np.ndarray,
    query_word_matrix,
    word_matrix,
    index: int,
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    svd: TruncatedSVD | None,
    component_labels: list[str],
    query_axis_indices: np.ndarray,
    query_axis_labels: list[str],
) -> dict:
    # convert one ranked flower into the response shape expected by the frontend
    matched_terms = _merge_keyword_lists(
        [
            _extract_query_morphology_keywords(
                query,
                _collect_keyword_candidates(flower=flower),
                MAX_MATCHED_KEYWORDS,
            ),
            _top_feature_terms(
                # multiply the query and flower word vectors to find shared explanation terms.
                query_word_matrix.multiply(word_matrix[index]),
                word_vectorizer,
                MAX_MATCHED_KEYWORDS,
            ),
        ],
        MAX_MATCHED_KEYWORDS,
    )
    for matched_term in matched_terms:
        if matched_term["category"] == "semantic":
            matched_term["category"] = _keyword_category_for_flower(matched_term["keyword"], flower)
    matched_terms = _merge_keyword_lists([matched_terms], MAX_MATCHED_KEYWORDS)
    displayed_meanings = _select_display_texts(flower["meanings"], query, 2)
    displayed_occasions = _select_display_texts(flower["occasions"], query, 2)

    if not displayed_meanings and not displayed_occasions:
        # if short metadata is missing, use passages as fallback explanation text
        highlights = _top_passages(query, flower["passages"], word_vectorizer, char_vectorizer, svd)
        displayed_meanings = highlights[:2]
        displayed_occasions = highlights[2:]

    radar_chart = None
    if len(query_axis_labels) >= 3:
        flower_axis_display_values = _flower_axis_display_values(
            flower_lsa_vector,
            query_axis_indices,
            query_axis_labels,
            matched_terms,
        )
        radar_chart = build_latent_radar_chart(
            flower_lsa_vector,
            component_labels,
            profile_kind="flower",
            axis_indices=query_axis_indices,
            axis_labels=query_axis_labels,
            axis_values=flower_axis_display_values,
        )

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
        "latent_radar_chart": None if radar_chart is None else radar_chart["image_data_url"],
        "latent_radar_axes": [] if radar_chart is None else radar_chart["axis_labels"],
        "image_url": flower.get("image_url"),
    }


# ADDED THIS BECAUSE I PLAN TO REUSE THE LOGIC HERE FOR MY BACKEND FOR THE 3D VISUALIZATION
def get_flower_vectors(scientific_names: list[str]) -> dict:
    """
    Public interface for other modules that need flower data.
    Returns everything update_bouquet needs without exposing internals.
    """
    flowers, _, _, _, _, _, lsa_matrix, _ = _load_model()
    
    sci_to_index = {
        _normalize(f["scientific_name"]): i
        for i, f in enumerate(flowers)
    }
    
    result = []
    for name in scientific_names:
        key = _normalize(name)
        if key in sci_to_index:
            i = sci_to_index[key]
            result.append({
                "index": i,
                "name": flowers[i]["name"],
                "scientific_name": flowers[i]["scientific_name"],
                "meanings": flowers[i]["meanings"],
                "colors": flowers[i]["colors"],
                "maintenance": flowers[i]["maintenance"],
                "plant_types": flowers[i]["plant_types"],
                "occasions": flowers[i]["occasions"],
                "lsa_vector": lsa_matrix[i],
            })
    
    return {
        "flowers": result,
        "total_flower_count": len(flowers),
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

    flowers, word_vectorizer, char_vectorizer, word_matrix, _, svd, lsa_matrix, component_labels = _load_model()
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
    query_breakdown_keywords = _build_query_breakdown_keywords(
        query,
        query_word_matrix,
        word_vectorizer,
        flowers,
        MAX_QUERY_KEYWORDS,
    )
    if not positive_scores:
        # even if no flower matches, we can still expose important query terms
        return _empty_response(query, query_breakdown_keywords)

    top_score = positive_scores[0]
    query_axes = _select_query_latent_axes(
        query_lsa[0],
        component_labels,
        query_breakdown_keywords,
        svd,
        word_vectorizer,
    )
    query_radar_chart = None
    query_axis_indices = np.asarray([], dtype=int)
    query_axis_labels: list[str] = []
    if query_axes is not None:
        query_axis_indices = query_axes["axis_indices"]
        query_axis_labels = _relabel_query_axes(
            query_axis_indices,
            query_axes["axis_labels"],
            query_breakdown_keywords,
            svd,
            word_vectorizer,
        )
        if ENABLE_QUERY_LATENT_DEBUG:
            _print_query_latent_dimensions(
                query,
                query_lsa[0],
                query_axis_indices,
                query_axis_labels,
            )
        query_axis_display_values = _query_axis_display_values(
            query_lsa[0],
            query_axis_indices,
            query_axis_labels,
            query_breakdown_keywords,
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
        # normalize each shown score relative to the top result for this query.
        suggestions.append(
            _build_suggestion(
                flowers[index],
                query,
                similarity,
                top_score,
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
        )

    # FINAALLLLYYYYY, WE CAN RETURN THE SUGGESTIONS OHHHH MY LAWWWWD THERE HAS TO BE AN EASIER WAY TO DO THIS
    return {
        "query": query,
        "keywords_used": query_breakdown_keywords,
        "query_latent_radar_chart": None if query_radar_chart is None else query_radar_chart["image_data_url"],
        "query_latent_radar_axes": [] if query_radar_chart is None else query_radar_chart["axis_labels"],
        "suggestions": suggestions,
    }
