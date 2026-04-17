"""
Shared utilities for recommender scripts: image URL resolution and CSV text splits.

Keeps a single implementation for local image matching, remote image_map fallbacks,
and semicolon/newline splitting of meaning cells used across TF-IDF & SVD version.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
FLOWER_IMAGE_DIR = _REPO_ROOT / "data_scraping" / "flower_images"
_IMAGE_MAP_PATH = _REPO_ROOT / "data_scraping" / "image_map.json"

try:
    _IMAGE_MAP: dict[str, str] = json.loads(_IMAGE_MAP_PATH.read_text(encoding="utf-8"))
except Exception:
    _IMAGE_MAP = {}

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
_IMAGE_ALIAS_TARGETS = {
    "arum lily": "calla lily",
    "pelargonium": "geranium",
    "zantedeschia": "calla lily",
}
_IMAGE_GENERIC_TOKENS = frozenset({"flower", "flowers", "meaning", "meanings"})
_COLOR_FALLBACK_IMAGE_FILENAMES = {
    "blue": "blue-flowers-meaning.jpg",
    "pink": "pink-flowers-meaning.jpg",
    "purple": "purple-flowers-meaning.jpg",
    "white": "white-flowers.jpg",
    "yellow": "yellow-flowers-meaning.jpg",
}
_GENERIC_FLOWER_IMAGE_FILENAME = "10-most-beautiful-flowers.jpg"

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

# token pattern because we need words for matching/displaying
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


# START OF TEXT CLEANUP HELPERS

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower()).strip()


def normalize_image_key(text: str) -> str:
    """
    Normalize image slugs with a small typo fix for scraped filenames.
    """
    return _normalize(text).replace("lilly", "lily")


def is_thematic_slug(slug: str) -> bool:
    """
    True if a filename stem looks like a category page rather than one flower.
    """
    return slug.endswith("flowers") or slug.startswith(THEMATIC_PREFIXES)


def _tokenize_for_image(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(_normalize(text))


def split_meaning_cell(value: str) -> list[str]:
    """
    Split a CSV meaning/occasion cell on semicolons and newlines (prototype3
    _split_meaning_chunks behavior).
    """
    parts = re.split(r"[;\n]+", value or "")
    return [part.strip(" .:") for part in parts if part.strip(" .:")]


def _load_local_image_index(include_thematic: bool = True) -> tuple[dict[str, str], set[str]]:
    try:
        image_map: dict[str, str] = {}
        filenames: set[str] = set()
        for path in FLOWER_IMAGE_DIR.iterdir():
            if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            if not include_thematic and is_thematic_slug(path.stem):
                continue
            normalized_key = normalize_image_key(path.stem)
            if normalized_key:
                image_map[normalized_key] = path.name
            filenames.add(path.name)
        return image_map, filenames
    except Exception:
        return {}, set()


_, _LOCAL_IMAGE_FILENAMES = _load_local_image_index(include_thematic=True)
_LOCAL_FLOWER_IMAGE_FILES, _ = _load_local_image_index(include_thematic=False)


def matching_image_candidates(flower: dict) -> list[tuple[str, str]]:
    """
    Ordered keys to try against local files and image_map (name, aliases,
    scientific name, plant types).
    """
    candidates: list[tuple[str, str]] = []
    name_key = normalize_image_key(flower.get("name", "") or "")
    if name_key:
        candidates.append(("name", name_key))
    for alias in flower.get("aliases") or []:
        ak = normalize_image_key(alias)
        if ak:
            candidates.append(("alias", ak))
    sci = flower.get("scientific_name")
    if sci:
        sk = normalize_image_key(sci)
        if sk:
            candidates.append(("scientific", sk))
    for pt in flower.get("plant_types") or []:
        pk = normalize_image_key(pt)
        if pk:
            candidates.append(("plant_type", pk))
    return candidates


def _image_match_score(candidate: str, local_key: str) -> int:
    candidate_tokens = [t for t in _tokenize_for_image(candidate) if t not in _IMAGE_GENERIC_TOKENS]
    local_tokens = [t for t in _tokenize_for_image(local_key) if t not in _IMAGE_GENERIC_TOKENS]
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
    normalized_candidate = normalize_image_key(candidate)
    if not normalized_candidate:
        return None

    candidates = [normalized_candidate]
    alias_target = _IMAGE_ALIAS_TARGETS.get(normalized_candidate)
    if alias_target:
        candidates.append(normalize_image_key(alias_target))

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


def resolve_flower_image_url(flower: dict) -> str | None:
    """
    Local tolerant filename match, then image_map.json, then color/generic fallbacks.
    """
    for _, candidate in matching_image_candidates(flower):
        filename = _best_local_image_filename(candidate)
        if filename:
            return f"/api/flower-images/{filename}"

        for remote_key, remote_url in _IMAGE_MAP.items():
            normalized_remote_key = normalize_image_key(remote_key)
            if (
                candidate == normalized_remote_key
                or candidate in normalized_remote_key
                or normalized_remote_key in candidate
            ):
                return remote_url

    return _fallback_image_url_from_metadata(flower)
