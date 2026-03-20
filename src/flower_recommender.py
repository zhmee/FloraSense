import csv
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

# the flower spreadsheet. because everything's on one csv...
DATA_FILE = Path(__file__).resolve().parent / "data" / "merged.csv"
# lowercase only and punctuation gone lets go
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
# words that show up in queries just to contribute absolutely nothing
STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "best", "by", "for", "from", "i",
    "in", "is", "it", "like", "me", "my", "myself", "of", "or", "our",
    "ours", "show", "something", "that", "the", "to", "want", "we", "with",
    # verb connectors that appear in queries as structure words, not content
    "mean", "means", "meaning", "meanings", "called", "known", "named",
    # generic flower prose filler
    "flower", "flowers", "bloom", "blooms", "plant", "plants",
}

# hand-tuned weights for each signal type
# yes this is a little vibes-based, but at least it's written down
BASE_WEIGHTS = {
    "common_name": 9.0,
    "scientific_name": 7.5,
    "color": 4.5,
    "maintenance": 4.0,
    "plant_type": 3.5,
    "meaning_phrase": 5.5,
    "meaning_token": 2.2,
}

# nicer labels for the frontend so it doesn't dump backend internals on people
CATEGORY_LABELS = {
    "common_name": "name",
    "scientific_name": "scientific name",
    "color": "color",
    "maintenance": "maintenance",
    "plant_type": "plant type",
    "meaning_phrase": "meaning",
    "meaning_token": "meaning",
}


def normalize_text(value):
    # lowercase everything and remove punctuation because life is easier that way
    return re.sub(r"[^a-z0-9\s]+", " ", value.lower()).strip()


def tokenize(value):
    # turn text into tokens so we have control (hopefully)
    return TOKEN_PATTERN.findall(normalize_text(value))


def split_list_field(value):
    # some csv cells decided one value was not enough so here's how we deal with it
    return [part.strip() for part in value.split(",") if part.strip()]


def split_meanings(value):
    # meanings are semicolon-separated, so split it up
    return [part.strip() for part in value.split(";") if part.strip()]


def normalize_plant_type(value):
    # make singular/plural plant types match so the query does not break the entire app
    lowered = value.lower().strip()
    aliases = {lowered}
    if lowered.endswith("ies") and len(lowered) > 3:
        aliases.add(lowered[:-3] + "y")
    elif lowered.endswith("s") and len(lowered) > 3:
        aliases.add(lowered[:-1])
    return aliases


def phrase_in_query(term, normalized_query, query_tokens):
    # multi-word terms have to show up as an actual phrase
    # single words get the easy route because they're already perfect the way they are
    if " " in term:
        return f" {term} " in f" {normalized_query} "
    return term in query_tokens


def build_term_variants(entry):
    # build every searchable term we can wring out of one flower entry
    terms = []
    terms.append((normalize_text(entry["common_name"]), "common_name"))
    terms.append((normalize_text(entry["scientific_name"]), "scientific_name"))

    for color in entry["colors"]:
        terms.append((normalize_text(color), "color"))

    for level in entry["maintenance"]:
        terms.append((normalize_text(level), "maintenance"))

    for plant_type in entry["plant_types"]:
        for alias in normalize_plant_type(plant_type):
            terms.append((alias, "plant_type"))

    for meaning in entry["meanings"]:
        normalized_meaning = normalize_text(meaning)
        terms.append((normalized_meaning, "meaning_phrase"))
        for token in tokenize(meaning):
            if len(token) > 2 and token not in STOPWORDS:
                terms.append((token, "meaning_token"))

    return terms


@lru_cache(maxsize=1)
def load_flower_catalog():
    # cache this so we do not reread the csv every request 
    grouped_entries = {}

    with DATA_FILE.open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            common_name = row.get("name", "").strip()
            scientific_name = row.get("scientific_name", "").strip()
            display_name = common_name or scientific_name or "Unknown flower"
            flower_key = normalize_text(common_name or scientific_name)

            if flower_key not in grouped_entries:
                grouped_entries[flower_key] = {
                    "key": flower_key,
                    "name": display_name,
                    "scientific_name": scientific_name or "Unknown",
                    "colors": set(),
                    "plant_types": set(),
                    "maintenance": set(),
                    "meanings": set(),
                }

            entry = grouped_entries[flower_key]
            if row.get("color", "").strip():
                entry["colors"].add(row["color"].strip())
            if row.get("maintenance", "").strip():
                entry["maintenance"].add(row["maintenance"].strip())

            for plant_type in split_list_field(row.get("planttype", "")):
                # one cell, many plant types... pain
                entry["plant_types"].add(plant_type)

            for meaning in split_meanings(row.get("meaning", "")):
                # keep full meaning phrases because single tokens alone were not cutting it
                entry["meanings"].add(meaning)

    flowers = []
    # term -> every flower/category combo that contains that term
    keyword_index = defaultdict(list)

    for entry in grouped_entries.values():
        # sort sets into lists so the api output stops behaving like a lucky accident
        flower = {
            "key": entry["key"],
            "name": entry["name"],
            "scientific_name": entry["scientific_name"],
            "colors": sorted(entry["colors"]),
            "plant_types": sorted(entry["plant_types"]),
            "maintenance": sorted(entry["maintenance"]),
            "meanings": sorted(entry["meanings"]),
        }
        searchable_tokens = set()
        # flatten everything into tokens for the fallback path when the nicer matches fail
        for field in [flower["name"], flower["scientific_name"], *flower["colors"], *flower["plant_types"], *flower["maintenance"], *flower["meanings"]]:
            searchable_tokens.update(tokenize(field))
        flower["searchable_tokens"] = searchable_tokens
        flowers.append(flower)

    # do not double-count the same term/category/flower combo (I SPENT AN HOUR FIGURING OUT WHY I WAS GETTIN 300%+ MATCHES)
    seen_index_entries = set()

    for flower in flowers:
        for term, category in build_term_variants({
            "common_name": flower["name"],
            "scientific_name": flower["scientific_name"],
            "colors": flower["colors"],
            "plant_types": flower["plant_types"],
            "maintenance": flower["maintenance"],
            "meanings": flower["meanings"],
        }):
            if not term or term in STOPWORDS:
                continue
            index_key = (term, category, flower["key"])
            if index_key in seen_index_entries:
                continue
            seen_index_entries.add(index_key)
            # each term keeps receipts for where it showed up
            keyword_index[term].append({
                "flower_key": flower["key"],
                "category": category,
            })

    return flowers, keyword_index


def score_contribution(category, match_count):
    # start with the category's base weight
    base_weight = BASE_WEIGHTS[category]
    # common terms get nerfed a bit so rare matches can stop being ignored
    rarity_adjustment = 1 / math.sqrt(match_count)
    return round(base_weight * rarity_adjustment, 2)


def recommend_flowers(query, limit=5):
    # clean up the raw query before we start pretending we understand language
    normalized_query = normalize_text(query or "")
    query_tokens = set(tokenize(query or ""))

    if not normalized_query:
        # empty query, empty results... obvs
        return {
            "query": query,
            "keywords_used": [],
            "suggestions": [],
        }

    flowers, keyword_index = load_flower_catalog()
    # these are the stronger matches we actually found in the query
    query_terms = []
    seen_terms = set()

    # longer terms first, because "low maintenance" should not lose to just "low" for no reason
    for term in sorted(keyword_index.keys(), key=lambda value: (-value.count(" "), -len(value), value)):
        if term in seen_terms:
            continue
        if phrase_in_query(term, normalized_query, query_tokens):
            seen_terms.add(term)
            descriptors = keyword_index[term]
            category = descriptors[0]["category"]
            # score the term using its category weight plus the rarity adjustment
            query_terms.append({
                "keyword": term,
                "category": CATEGORY_LABELS[category],
                "score": score_contribution(category, len(descriptors)),
            })

    # leftover tokens still get a shot, just with much less authority
    fallback_tokens = [
        token for token in tokenize(query or "")
        if token not in STOPWORDS and token not in seen_terms and len(token) > 2
    ]

    # running score per flower
    scores = defaultdict(float)
    # score receipts for the frontend
    matched_keywords = defaultdict(list)
    # track how many different signal types matched
    matched_categories = defaultdict(set)

    for term in query_terms:
        raw_descriptors = keyword_index[term["keyword"]]
        contribution = term["score"]
        for descriptor in raw_descriptors:
            flower_key = descriptor["flower_key"]
            # every flower containing the term gets that weighted bump (nice)
            scores[flower_key] += contribution
            matched_categories[flower_key].add(term["category"])
            matched_keywords[flower_key].append({
                "keyword": term["keyword"],
                "category": term["category"],
                "score": contribution,
            })

    for flower in flowers:
        overlaps = sorted(set(fallback_tokens) & flower["searchable_tokens"])
        if not overlaps:
            continue
        for token in overlaps:
            # fallback overlap is weaker on purpose because accidental overlap should calm down.
            scores[flower["key"]] += 1.1
            matched_categories[flower["key"]].add("text")
            matched_keywords[flower["key"]].append({
                "keyword": token,
                "category": "text",
                "score": 1.1,
            })

    suggestions = []
    for flower in flowers:
        flower_score = scores.get(flower["key"], 0)
        if flower_score <= 0:
            continue

        # small bonus for matching across different categories instead of brute-forcing one lane.
        diversity_bonus = max(len(matched_categories[flower["key"]]) - 1, 0) * 0.75
        total_score = round(flower_score + diversity_bonus, 2)
        # put the strongest reasons first so the ui can act like it had a plan all along.
        reasons = sorted(
            matched_keywords[flower["key"]],
            key=lambda item: (-item["score"], item["keyword"]),
        )

        # final payload for one flower suggestion.
        suggestions.append({
            "name": flower["name"],
            "scientific_name": flower["scientific_name"],
            "colors": flower["colors"],
            "plant_types": flower["plant_types"],
            "maintenance": flower["maintenance"],
            "meanings": flower["meanings"],
            "score": total_score,
            "matched_keywords": reasons,
        })

    # best flowers first.. stunning
    suggestions.sort(key=lambda item: (-item["score"], item["name"].lower()))
    # same deal for matched query keywords
    keywords_used = sorted(query_terms, key=lambda item: (-item["score"], item["keyword"]))

    return {
        "query": query,
        "keywords_used": keywords_used,
        "suggestions": suggestions[:limit],
    }
