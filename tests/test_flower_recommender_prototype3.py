from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import flower_recommender as baseline_recommender
import flower_autocomplete
import flower_recommender_prototype3 as prototype3


SEMANTIC_CASES = [
    {
        "query": "a flower for secret love",
        "expected": {"gardenia"},
        "max_rank": 1,
    },
    {
        "query": "something for remembrance of a missing friend",
        "expected": {"zinnia"},
        "max_rank": 3,
    },
    {
        "query": "a flower for rebirth and enlightenment",
        "expected": {"lotus"},
        "max_rank": 1,
    },
    {
        "query": "a flower that represents strength and courage",
        "expected": {"gladiolus", "snapdragon", "protea"},
        "max_rank": 1,
    },
]

SMOKE_QUERY_CASES = [
    {
        "query": "gardenia",
        "expected": {"gardenia"},
        "max_rank": 1,
    },
    {
        "query": "myosotis",
        "expected": {"forget me not"},
        "max_rank": 3,
    },
    {
        "query": "wedding anniversary flowers",
        "expected": set(),
        "max_rank": None,
    },
    {
        "query": "low maintenance pink flowers",
        "expected": set(),
        "max_rank": None,
        "required_color": "pink",
        "required_maintenance": "low",
    },
]


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum() or ch.isspace()).strip()


def _top_names(result: dict, limit: int) -> list[str]:
    return [item["name"] for item in result.get("suggestions", [])[:limit]]


def _contains_expected_name(names: list[str], expected: set[str]) -> bool:
    normalized_names = [_normalize_name(name) for name in names]
    for target in expected:
        normalized_target = _normalize_name(target)
        for candidate in normalized_names:
            if normalized_target in candidate or candidate in normalized_target:
                return True
    return False


def _best_rank(names: list[str], expected: set[str]) -> int | None:
    normalized_names = [_normalize_name(name) for name in names]
    normalized_expected = {_normalize_name(name) for name in expected}

    for index, candidate in enumerate(normalized_names, start=1):
        for target in normalized_expected:
            if target in candidate or candidate in target:
                return index
    return None


def _suggestions_contain_value(result: dict, field: str, target: str) -> bool:
    normalized_target = _normalize_name(target)
    for suggestion in result.get("suggestions", []):
        values = suggestion.get(field, [])
        for value in values:
            normalized_value = _normalize_name(value)
            if normalized_target in normalized_value or normalized_value in normalized_target:
                return True
    return False


def _ranking_metrics(recommender, limit: int = 5) -> dict:
    ranks = []

    for case in SEMANTIC_CASES:
        result = recommender.recommend_flowers(case["query"], limit=limit)
        rank = _best_rank(_top_names(result, limit), case["expected"])
        ranks.append(rank)

    total = len(ranks)
    return {
        "hit_at_1": sum(rank == 1 for rank in ranks),
        "hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks),
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks),
        "mrr": sum(0.0 if rank is None else 1.0 / rank for rank in ranks),
        "case_count": total,
        "ranks": ranks,
    }


def _is_query_shaped_suggestion(suggestion: str) -> bool:
    normalized = _normalize_name(suggestion)
    return (
        normalized.endswith(" flowers")
        or normalized.startswith("flowers for ")
        or normalized.startswith("a flower for ")
    )


class FlowerRecommenderPrototype3Tests(unittest.TestCase):
    def test_autocomplete_returns_svd_backed_suggestions(self):
        result = flower_autocomplete.autocomplete_queries("low ma")

        self.assertEqual(result["query"], "low ma")
        self.assertTrue(result["suggestions"])
        self.assertIn("low maintenance flowers", result["suggestions"])
        self.assertEqual(result["suggestions"], [suggestion.lower() for suggestion in result["suggestions"]])
        self.assertTrue(all(_is_query_shaped_suggestion(suggestion) for suggestion in result["suggestions"]))

    def test_autocomplete_filters_noisy_single_word_flowers_for_prompts(self):
        result = flower_autocomplete.autocomplete_queries("flowers for")

        self.assertTrue(result["suggestions"])
        self.assertNotIn("flowers for gay", result["suggestions"])
        self.assertTrue(all(_is_query_shaped_suggestion(suggestion) for suggestion in result["suggestions"]))

    def test_autocomplete_does_not_return_specific_flower_names(self):
        result = flower_autocomplete.autocomplete_queries("gar")

        self.assertTrue(all(_is_query_shaped_suggestion(suggestion) for suggestion in result["suggestions"]))
        self.assertNotIn("gardenia", result["suggestions"])

    def test_autocomplete_keeps_full_query_prefix_for_partial_last_word(self):
        result = flower_autocomplete.autocomplete_queries("a flower for c")

        self.assertTrue(result["suggestions"])
        self.assertTrue(
            all(suggestion.startswith("a flower for c") for suggestion in result["suggestions"]),
            msg=f"Expected autocomplete to preserve the typed prefix, got {result['suggestions']}",
        )

    def test_model_info_uses_scraped_corpus_and_svd(self):
        info = prototype3.model_info()
        corpus_dir = Path(info["corpus_dir"])

        self.assertEqual(info["retrieval_mode"], "svd_only")
        self.assertEqual(corpus_dir.name, "flower_texts")
        self.assertEqual(corpus_dir.parent.name, "data_scraping")
        self.assertGreater(info["document_count"], 0)
        self.assertGreater(info["feature_count"], 0)
        self.assertGreater(info["svd_components"], 0)

    def test_matched_keywords_use_specific_metadata_categories_when_possible(self):
        result = prototype3.recommend_flowers("low maintenance pink flowers", limit=5)

        self.assertTrue(result["suggestions"])
        categories = {
            match["category"]
            for suggestion in result["suggestions"]
            for match in suggestion.get("matched_keywords", [])
        }
        self.assertIn("color", categories)
        self.assertIn("maintenance", categories)

    def test_query_breakdown_uses_specific_metadata_categories_when_possible(self):
        result = prototype3.recommend_flowers("white flowers for gratitude", limit=5)

        keyword_categories = {
            item["keyword"]: item["category"]
            for item in result["keywords_used"]
        }
        self.assertEqual(keyword_categories.get("white"), "color")

    def test_semantic_queries_return_results(self):
        for case in SEMANTIC_CASES:
            with self.subTest(query=case["query"]):
                result = prototype3.recommend_flowers(case["query"], limit=5)
                self.assertTrue(result["suggestions"], msg=f"No suggestions for: {case['query']}")

    def test_expected_flowers_appear_in_top_five(self):
        for case in SEMANTIC_CASES:
            with self.subTest(query=case["query"]):
                result = prototype3.recommend_flowers(case["query"], limit=5)
                top_names = _top_names(result, 5)
                self.assertTrue(
                    _contains_expected_name(top_names, case["expected"]),
                    msg=f"Expected one of {sorted(case['expected'])} in top 5 for {case['query']}, got {top_names}",
                )

    def test_expected_flowers_rank_near_the_top(self):
        for case in SEMANTIC_CASES:
            with self.subTest(query=case["query"]):
                result = prototype3.recommend_flowers(case["query"], limit=5)
                top_names = _top_names(result, 5)
                rank = _best_rank(top_names, case["expected"])
                self.assertIsNotNone(
                    rank,
                    msg=f"Expected one of {sorted(case['expected'])} in ranked results for {case['query']}, got {top_names}",
                )
                self.assertLessEqual(
                    rank,
                    case["max_rank"],
                    msg=f"Expected one of {sorted(case['expected'])} by rank {case['max_rank']} for {case['query']}, got rank {rank} from {top_names}",
                )

    def test_prototype3_handles_diverse_query_types(self):
        for case in SMOKE_QUERY_CASES:
            with self.subTest(query=case["query"]):
                result = prototype3.recommend_flowers(case["query"], limit=5)
                top_names = _top_names(result, 5)

                self.assertTrue(top_names, msg=f"No suggestions returned for {case['query']}")

                if case["expected"]:
                    rank = _best_rank(top_names, case["expected"])
                    self.assertIsNotNone(
                        rank,
                        msg=f"Expected one of {sorted(case['expected'])} in top results for {case['query']}, got {top_names}",
                    )
                    self.assertLessEqual(
                        rank,
                        case["max_rank"],
                        msg=f"Expected one of {sorted(case['expected'])} by rank {case['max_rank']} for {case['query']}, got rank {rank} from {top_names}",
                    )

                if "required_color" in case:
                    self.assertTrue(
                        _suggestions_contain_value(result, "colors", case["required_color"]),
                        msg=f"Expected at least one suggested flower with color {case['required_color']} for {case['query']}",
                    )

                if "required_maintenance" in case:
                    self.assertTrue(
                        _suggestions_contain_value(result, "maintenance", case["required_maintenance"]),
                        msg=f"Expected at least one suggested flower with maintenance {case['required_maintenance']} for {case['query']}",
                    )

    def test_empty_and_whitespace_queries_return_empty_results(self):
        for query in ["", "   "]:
            with self.subTest(query=repr(query)):
                result = prototype3.recommend_flowers(query)
                self.assertEqual(result["suggestions"], [])
                self.assertEqual(result["keywords_used"], [])

    def test_semantic_ranking_metrics_match_or_beat_baseline(self):
        prototype_metrics = _ranking_metrics(prototype3, limit=5)
        baseline_metrics = _ranking_metrics(baseline_recommender, limit=5)

        print(
            "\nsemantic ranking benchmark:",
            {
                "prototype3": prototype_metrics,
                "baseline": baseline_metrics,
            },
        )

        self.assertGreaterEqual(
            prototype_metrics["hit_at_1"],
            baseline_metrics["hit_at_1"],
            msg="Prototype 3 should match or beat the baseline on hit@1.",
        )
        self.assertGreaterEqual(
            prototype_metrics["hit_at_3"],
            baseline_metrics["hit_at_3"],
            msg="Prototype 3 should match or beat the baseline on hit@3.",
        )
        self.assertGreaterEqual(
            prototype_metrics["hit_at_5"],
            baseline_metrics["hit_at_5"],
            msg="Prototype 3 should match or beat the baseline on hit@5.",
        )
        self.assertGreaterEqual(
            prototype_metrics["mrr"],
            baseline_metrics["mrr"],
            msg="Prototype 3 should match or beat the baseline on mean reciprocal rank.",
        )


if __name__ == "__main__":
    unittest.main()
