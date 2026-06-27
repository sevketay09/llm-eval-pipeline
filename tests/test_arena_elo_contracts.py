"""Contract tests for analysis.arena_elo — offline, no real LLM."""
import json
import math
import unittest

from analysis.arena_elo import (
    build_leaderboard,
    compute_bradley_terry,
    compute_elo,
    normalize_matches,
    summarize_arena,
)


def _matches(pairs):
    """Helper to build match list from (model_a, model_b, winner) tuples."""
    return [{"model_a": a, "model_b": b, "winner": w} for a, b, w in pairs]


class ComputeEloContractTests(unittest.TestCase):
    """Contract tests for compute_elo."""

    def test_winner_gains_rating(self):
        """Model that wins should gain rating, loser should lose."""
        matches = _matches([("model_a", "model_b", "model_a")])
        ratings = compute_elo(matches)

        self.assertGreater(ratings["model_a"], 1500.0)
        self.assertLess(ratings["model_b"], 1500.0)

    def test_tie_moves_toward_equal(self):
        """Equal strength models that tie should stay near initial rating."""
        matches = _matches([("model_a", "model_b", "tie")])
        ratings = compute_elo(matches)

        # Ratings should be close to initial (within K/2 for single match)
        self.assertAlmostEqual(ratings["model_a"], 1500.0, delta=16.0)
        self.assertAlmostEqual(ratings["model_b"], 1500.0, delta=16.0)

    def test_all_models_present(self):
        """All models in matches should appear in result."""
        matches = _matches([
            ("model_a", "model_b", "model_a"),
            ("model_b", "model_c", "model_c"),
        ])
        ratings = compute_elo(matches)

        self.assertIn("model_a", ratings)
        self.assertIn("model_b", ratings)
        self.assertIn("model_c", ratings)

    def test_initial_rating_respected(self):
        """Custom initial rating should be used."""
        matches = _matches([("model_a", "model_b", "model_a")])
        ratings = compute_elo(matches, initial_rating=1000.0)

        # After one match, both should be near 1000 (but slightly different)
        self.assertGreater(ratings["model_a"], 1000.0)
        self.assertLess(ratings["model_b"], 1000.0)


class ComputeBradleyTerryContractTests(unittest.TestCase):
    """Contract tests for compute_bradley_terry."""

    def test_dominant_model_higher_score(self):
        """Model winning most games should have higher score."""
        # model_a wins 8/10 games
        matches = _matches([
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_b", "model_b"),
            ("model_a", "model_b", "model_b"),
        ])
        scores = compute_bradley_terry(matches)

        self.assertGreater(scores["model_a"], scores["model_b"])

    def test_all_models_present(self):
        """All models should appear in result."""
        matches = _matches([
            ("model_a", "model_b", "model_a"),
            ("model_b", "model_c", "model_c"),
        ])
        scores = compute_bradley_terry(matches)

        self.assertIn("model_a", scores)
        self.assertIn("model_b", scores)
        self.assertIn("model_c", scores)

    def test_single_model_returns_one(self):
        """Single model should return score of 1.0."""
        matches = [{"model_a": "model_x", "model_b": "model_x", "winner": "tie"}]
        scores = compute_bradley_terry(matches)

        self.assertEqual(scores.get("model_x"), 1.0)


class BuildLeaderboardContractTests(unittest.TestCase):
    """Contract tests for build_leaderboard."""

    def test_sorted_descending(self):
        """Leaderboard should be sorted by rating descending."""
        matches = _matches([
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_c", "model_a"),
            ("model_b", "model_c", "model_c"),
        ])
        leaderboard = build_leaderboard(matches)

        for i in range(len(leaderboard) - 1):
            self.assertGreaterEqual(leaderboard[i]["rating"], leaderboard[i+1]["rating"])

    def test_ranks_sequential(self):
        """Ranks should be 1, 2, 3, ..."""
        matches = _matches([
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_c", "model_a"),
        ])
        leaderboard = build_leaderboard(matches)

        for i, entry in enumerate(leaderboard, 1):
            self.assertEqual(entry["rank"], i)

    def test_win_loss_counts_correct(self):
        """Win/loss counts should match match results."""
        matches = _matches([
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_c", "model_a"),
            ("model_a", "model_b", "model_a"),
        ])
        leaderboard = build_leaderboard(matches)

        model_a_entry = [e for e in leaderboard if e["model"] == "model_a"][0]
        self.assertEqual(model_a_entry["wins"], 3)
        self.assertEqual(model_a_entry["losses"], 0)

    def test_win_rate_computed(self):
        """Win rate should be wins / (wins + losses)."""
        matches = _matches([
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_b", "model_b"),
        ])
        leaderboard = build_leaderboard(matches)

        model_a_entry = [e for e in leaderboard if e["model"] == "model_a"][0]
        self.assertAlmostEqual(model_a_entry["win_rate"], 0.75, places=2)


class SummarizeArenaContractTests(unittest.TestCase):
    """Contract tests for summarize_arena."""

    def test_top_level_keys(self):
        """Summary should have required top-level keys."""
        matches = _matches([
            ("model_a", "model_b", "model_a"),
        ])
        summary = summarize_arena(matches)

        self.assertIn("total_matches", summary)
        self.assertIn("models", summary)
        self.assertIn("method", summary)
        self.assertIn("leaderboard", summary)
        self.assertIn("category_breakdown", summary)
        self.assertIn("head_to_head", summary)

    def test_total_matches_count(self):
        """total_matches should equal number of matches."""
        matches = _matches([
            ("model_a", "model_b", "model_a"),
            ("model_a", "model_c", "model_b"),
            ("model_b", "model_c", "model_c"),
            ("model_a", "model_b", "tie"),
            ("model_a", "model_b", "model_a"),
        ])
        summary = summarize_arena(matches)

        self.assertEqual(summary["total_matches"], 5)

    def test_category_breakdown_populated(self):
        """Category breakdown should be populated for matches with category."""
        matches = [
            {"model_a": "model_a", "model_b": "model_b", "winner": "model_a", "category": "cat1"},
            {"model_a": "model_a", "model_b": "model_b", "winner": "model_b", "category": "cat2"},
        ]
        summary = summarize_arena(matches)

        self.assertIn("cat1", summary["category_breakdown"])
        self.assertIn("cat2", summary["category_breakdown"])
        self.assertIn("model_a", summary["category_breakdown"]["cat1"])


class NormalizeMatchesContractTests(unittest.TestCase):
    """Contract tests for normalize_matches."""

    def test_flat_list_passthrough(self):
        """Flat list of matches should be returned as-is."""
        data = [
            {"model_a": "a", "model_b": "b", "winner": "model_a"},
            {"model_a": "b", "model_b": "c", "winner": "model_b"},
        ]
        result = normalize_matches(data)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["model_a"], "a")
        self.assertEqual(result[1]["model_b"], "c")

    def test_arena_output_expanded(self):
        """Arena output format should be expanded to flat matches."""
        data = {
            "model_a": "gpt-4o",
            "model_b": "qwen-32b",
            "results": [
                {"winner": "model_a", "category": "math"},
                {"winner": "model_b"},
                {"winner": "tie"},
            ]
        }
        result = normalize_matches(data)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["model_a"], "gpt-4o")
        self.assertEqual(result[0]["model_b"], "qwen-32b")
        self.assertEqual(result[0]["winner"], "model_a")
        self.assertEqual(result[0]["category"], "math")
        self.assertEqual(result[1]["winner"], "model_b")
        self.assertNotIn("category", result[1])


if __name__ == "__main__":
    unittest.main()
