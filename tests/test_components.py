"""Unit tests for the refactored components.

Each test targets one module boundary so regressions are immediately attributable.
Run with: python -m pytest tests/
"""
from __future__ import annotations

import math
import unittest

# ---------------------------------------------------------------------------
# src/catalog.py
class CatalogTextTest(unittest.TestCase):
    def test_text_flattens_dict(self):
        from src.catalog import text
        self.assertEqual(text({"k": "v"}), "k v")

    def test_text_flattens_list(self):
        from src.catalog import text
        self.assertEqual(text(["a", "b"]), "a b")

    def test_text_none(self):
        from src.catalog import text
        self.assertEqual(text(None), "")

    def test_terms_drops_stopwords(self):
        from src.catalog import terms
        result = terms("I am looking for a cotton shirt")
        self.assertIn("cotton", result)
        self.assertIn("shirt", result)
        self.assertNotIn("for", result)
        self.assertNotIn("a", result)

    def test_terms_drops_single_chars(self):
        from src.catalog import terms
        self.assertNotIn("i", terms("i want x"))


# ---------------------------------------------------------------------------
# src/retrieval.py
class RRFTest(unittest.TestCase):
    def test_rrf_union(self):
        from src.retrieval import rrf
        result = rrf(["A", "B"], ["B", "C"], 0.5, top_n=10)
        self.assertEqual(set(result), {"A", "B", "C"})

    def test_rrf_top_n_limit(self):
        from src.retrieval import rrf
        result = rrf(["A", "B", "C"], ["D", "E"], 0.5, top_n=3)
        self.assertEqual(len(result), 3)

    def test_rrf_shared_item_scores_higher(self):
        from src.retrieval import rrf
        result = rrf(["A", "B"], ["A", "C"], 1.0, top_n=10)
        self.assertEqual(result[0], "A")  # A appears in both lists → highest score

    def test_vector_weight_buying(self):
        from src.retrieval import vector_weight
        from src.config import BUYING_VECTOR_WEIGHT
        w = vector_weight(1.0, use_intent_routing=True, use_confidence_routing=True)
        self.assertAlmostEqual(w, BUYING_VECTOR_WEIGHT, places=5)

    def test_vector_weight_browsing(self):
        from src.retrieval import vector_weight
        from src.config import BROWSING_VECTOR_WEIGHT
        w = vector_weight(0.0, use_intent_routing=True, use_confidence_routing=True)
        self.assertAlmostEqual(w, BROWSING_VECTOR_WEIGHT, places=5)

    def test_vector_weight_no_routing(self):
        from src.retrieval import vector_weight
        from src.config import VECTOR_WEIGHT
        w = vector_weight(0.9, use_intent_routing=False)
        self.assertAlmostEqual(w, VECTOR_WEIGHT, places=5)


# ---------------------------------------------------------------------------
# src/ranking.py
class CoverageRerankerTest(unittest.TestCase):
    def _make_reranker(self):
        from src.ranking import CoverageReranker
        catalog = {
            "A": {"title": "waterproof leather boot", "features": ["waterproof"],
                  "details": {}, "categories": [], "store": "", "description": []},
            "B": {"title": "canvas sneaker", "features": [], "details": {},
                  "categories": [], "store": "", "description": []},
        }
        return CoverageReranker(catalog)

    def test_verbatim_phrase_moves_to_top(self):
        cr = self._make_reranker()
        ordered, scores = cr.rerank_scored(["B", "A"], ["waterproof leather"])
        self.assertEqual(ordered[0], "A")
        self.assertGreater(scores["A"], scores["B"])

    def test_no_phrases_preserves_order(self):
        cr = self._make_reranker()
        ordered, scores = cr.rerank_scored(["B", "A"], [])
        self.assertEqual(ordered, ["B", "A"])
        self.assertEqual(scores, {})

    def test_doc_is_cached(self):
        cr = self._make_reranker()
        d1 = cr.doc("A")
        d2 = cr.doc("A")
        self.assertIs(d1, d2)  # same object → cache hit


class PersonalizerTest(unittest.TestCase):
    def test_popular_item_moves_up(self):
        from src.ranking import Personalizer
        catalog = {
            "A": {"title": "cotton shirt", "features": [], "categories": [],
                  "rating_number": 100},
            "B": {"title": "silk blouse", "features": [], "categories": [],
                  "rating_number": 10000},
        }
        p = Personalizer(catalog)
        result = p.rerank(["A", "B"], {}, strength=0.0)
        self.assertEqual(result[0], "B")


# ---------------------------------------------------------------------------
# src/dialogue.py
class IntentRouterTest(unittest.TestCase):
    def test_buying_phrase_scores_high(self):
        from src.dialogue import IntentRouter
        r = IntentRouter()
        self.assertGreater(r.score("a key requirement is leather", 8), 0.6)

    def test_browsing_phrase_scores_low(self):
        from src.dialogue import IntentRouter
        r = IntentRouter()
        self.assertLess(r.score("just browsing, not sure", 3), 0.4)

    def test_is_override(self):
        from src.dialogue import IntentRouter
        r = IntentRouter()
        self.assertTrue(r.is_override("Actually, ignore my earlier preference."))
        self.assertFalse(r.is_override("I need a leather jacket"))

    def test_label_boundaries(self):
        from src.dialogue import IntentRouter
        self.assertEqual(IntentRouter.label(0.7), "buying")
        self.assertEqual(IntentRouter.label(0.3), "browsing")
        self.assertEqual(IntentRouter.label(0.5), "mixed")


class ExtractConstraintsTest(unittest.TestCase):
    def test_extracts_after_marker(self):
        from src.dialogue import extract_constraints
        msg = "What I need is: waterproof; size 10."
        self.assertEqual(extract_constraints(msg), ["waterproof", "size 10"])

    def test_no_marker_returns_empty(self):
        from src.dialogue import extract_constraints
        self.assertEqual(extract_constraints("I want something warm"), [])

    def test_short_fragments_skipped(self):
        from src.dialogue import extract_constraints
        msg = "What I need is: ok; leather."
        # "ok" is 2 chars → stripped; "leather" kept
        self.assertIn("leather", extract_constraints(msg))


# ---------------------------------------------------------------------------
# src/understanding.py — category resolver
class CategoryResolverTest(unittest.TestCase):
    def test_handbag_resolves_to_bag(self):
        from src.understanding import resolve_category
        self.assertEqual(resolve_category("roomy fashion hobo handbag"), "bag")

    def test_leftmost_wins(self):
        from src.understanding import resolve_category
        # "handbag" appears before "wallet" in the text → bag wins
        self.assertEqual(resolve_category("nice handbag with a small wallet pocket"), "bag")

    def test_bra_resolves(self):
        from src.understanding import resolve_category
        self.assertEqual(resolve_category("sports bra"), "bra")

    def test_nightgown_resolves_to_sleepwear(self):
        from src.understanding import resolve_category
        self.assertEqual(resolve_category("nightgown for sleeping"), "sleepwear")

    def test_jersey_resolves_to_shirt(self):
        from src.understanding import resolve_category
        self.assertEqual(resolve_category("cycling short sleeve jersey"), "shirt")

    def test_scrubs_resolves_to_pants(self):
        from src.understanding import resolve_category
        self.assertEqual(resolve_category("iflex scrubs for women"), "pants")

    def test_none_when_no_match(self):
        from src.understanding import resolve_category
        self.assertIsNone(resolve_category("some random text with no apparel nouns"))


# ---------------------------------------------------------------------------
# src/agent.py — integration smoke test
class AgentSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        import tempfile
        from pathlib import Path
        cls._tmpdir = tempfile.TemporaryDirectory()
        catalog_path = Path(cls._tmpdir.name) / "catalog.jsonl"
        rows = [
            {"parent_asin": "A", "title": "waterproof leather hiking boot",
             "features": ["waterproof"], "details": {"size": "10"},
             "description": ["rugged boot"], "categories": ["Clothing", "Boots"],
             "store": "BootCo", "average_rating": 4.5, "rating_number": 500, "price": 89.0},
            {"parent_asin": "B", "title": "canvas sneaker",
             "features": [], "details": {}, "description": [],
             "categories": ["Clothing", "Shoes"], "store": "SneakerCo",
             "average_rating": 4.0, "rating_number": 100, "price": 49.0},
        ]
        catalog_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        from src.agent import Agent
        Agent.USE_VECTOR = False  # avoid pulling real cache/embeddings.npy in unit tests
        cls.agent = Agent(str(catalog_path))

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def setUp(self):
        self.agent.reset("sess", {"preference_tags": [], "summary": ""})

    def test_reset_creates_session(self):
        self.assertIn("sess", self.agent._sessions)

    def test_respond_returns_required_keys(self):
        resp = self.agent.respond("sess", "I need a waterproof boot", 1, 5)
        self.assertIn("message", resp)
        self.assertIn("ask_attribute", resp)
        self.assertIn("recommendations", resp)
        self.assertIn("usage", resp)

    def test_recommendations_are_valid_asins(self):
        resp = self.agent.respond("sess", "leather hiking boot", 1, 5)
        for rec in resp["recommendations"]:
            self.assertIn("parent_asin", rec)
            self.assertIn(rec["parent_asin"], self.agent.catalog)

    def test_coverage_reranker_lifts_target(self):
        # Constraint phrase is verbatim from product A's catalog text
        self.agent.respond("sess", "I need a boot", 1, 5)
        resp = self.agent.respond(
            "sess", "A key requirement is: waterproof leather hiking boot", 2, 5)
        asins = [r["parent_asin"] for r in resp["recommendations"]]
        self.assertEqual(asins[0], "A")

    def test_catalog_property(self):
        catalog = self.agent.catalog
        self.assertIn("A", catalog)
        self.assertEqual(catalog["A"]["title"], "waterproof leather hiking boot")

    def test_reset_clears_session(self):
        self.agent.respond("sess", "warm jacket", 1, 5)
        self.agent.reset("sess", {})
        st = self.agent._sessions["sess"]
        self.assertEqual(st.all_text, [])
        self.assertEqual(st.constraint_phrases, [])


if __name__ == "__main__":
    unittest.main()
