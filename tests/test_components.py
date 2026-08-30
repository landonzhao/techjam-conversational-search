"""Unit tests for the refactored components.

Each test targets one module boundary so regressions are immediately attributable.
Run with: python -m pytest tests/
"""
from __future__ import annotations

import math
import unittest
from collections import Counter

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

    def test_pool_override_is_independent_of_personalization(self):
        # Popularity/profile ablations must retain the normal candidate pool.
        self.agent.POOL_SIZE_OVERRIDE = 200
        self.agent.USE_PERSONALIZATION = False
        self.assertEqual(self.agent._pool_size(self.agent._sessions["sess"]), 200)
        self.agent.POOL_SIZE_OVERRIDE = None
        self.agent.USE_PERSONALIZATION = True


# ---------------------------------------------------------------------------
# src/ranking.py — discrimination floor gate (conditional retrieval floor)
class DiscriminationGateTest(unittest.TestCase):
    def _cov(self):
        from src.ranking import CoverageReranker
        cat = {
            "T":  {"title": "zephyr corduroy jacket", "features": "napped pile warm",
                   "rating_number": 10},
            "B1": {"title": "acme corduroy jacket", "features": "plain", "rating_number": 9000},
            "B2": {"title": "acme corduroy coat", "features": "plain", "rating_number": 9000},
            "X":  {"title": "unrelated sandal", "features": "beach", "rating_number": 5},
        }
        return CoverageReranker(cat)

    def test_discriminating_phrase_floor_off_target_leads(self):
        # A word only the target carries -> gate says informative -> floor OFF -> coverage wins.
        order, _ = self._cov().rerank_scored(
            ["B1", "B2", "T", "X"], ["zephyr", "napped pile"], retrieval_weight=2.0,
            informative_min=0.5, discrimination_pctl=0.9,
            suppress_pop_on_paraphrase=True, pop_blend=0.1)
        self.assertEqual(order[0], "T")

    def test_shared_anchor_floor_on_target_not_buried(self):
        # A word many rivals share -> gate says uninformative -> floor ON + pop suppressed ->
        # retrieval order preserved, so the target is not sunk under the 9000-rating brand-mates.
        order, _ = self._cov().rerank_scored(
            ["B1", "B2", "T", "X"], ["corduroy"], retrieval_weight=2.0,
            informative_min=0.5, discrimination_pctl=0.9,
            suppress_pop_on_paraphrase=True, pop_blend=0.1)
        self.assertLessEqual(order.index("T"), 2)

    def test_gate_off_is_unconditional_floor(self):
        # informative_min=0 -> legacy behaviour, no crash, order is a permutation of the input.
        order, _ = self._cov().rerank_scored(
            ["B1", "B2", "T", "X"], ["corduroy"], retrieval_weight=2.0, informative_min=0.0)
        self.assertEqual(set(order), {"B1", "B2", "T", "X"})


# ---------------------------------------------------------------------------
# src/ranking.py — NeedSatisfactionScorer (generalized coverage)
class SatisfactionScorerTest(unittest.TestCase):
    def _make(self, vector=None, sem_alpha=1.0):
        from src.ranking import CoverageReranker, NeedSatisfactionScorer
        cat = {
            "T": {"title": "corduroy jacket", "features": "napped pile warm"},
            "P": {"title": "plain cotton tee", "features": "basic"},
            "Q": {"title": "leather belt", "features": "buckle"},
        }
        return NeedSatisfactionScorer(CoverageReranker(cat), vector=vector, sem_alpha=sem_alpha)

    def test_lexical_match_ranks_target_first(self):
        # No vector -> pure lexical. Only T contains "corduroy" -> T leads.
        order, sat = self._make().rank(["P", "Q", "T"], ["corduroy jacket"])
        self.assertEqual(order[0], "T")
        self.assertGreater(sat["T"], sat["P"])

    def test_empty_phrases_preserve_order(self):
        order, _ = self._make().rank(["P", "Q", "T"], [])
        self.assertEqual(order, ["P", "Q", "T"])

    def test_semantic_rescues_paraphrase_without_lexical_overlap(self):
        # A phrase with NO lexical overlap + a stub vector that only T is similar to -> the
        # semantic term must lift T above the lexically-tied rivals.
        class StubVector:
            def phrase_similarity_matrix(self, phrases, asins):
                return {"T": [0.9], "P": [0.1], "Q": [0.05]}
        order, sat = self._make(vector=StubVector()).rank(
            ["P", "Q", "T"], ["ribbed velvety fabric"])
        self.assertEqual(order[0], "T")
        self.assertGreater(sat["T"], sat["P"])

    def test_sem_alpha_zero_ignores_semantic(self):
        # sem_alpha=0 disables the semantic term -> a stub vector cannot change the order; with no
        # lexical overlap all scores are 0 and the input order is preserved (== coverage behaviour).
        class StubVector:
            def phrase_similarity_matrix(self, phrases, asins):
                return {a: [0.9] for a in asins}
        order, sat = self._make(vector=StubVector(), sem_alpha=0.0).rank(
            ["P", "Q", "T"], ["ribbed velvety fabric"])
        self.assertEqual(order, ["P", "Q", "T"])
        self.assertEqual(max(sat.values()), 0.0)


class ScenarioMixTest(unittest.TestCase):
    def test_largest_remainder_matches_official_mix(self):
        from scripts.scenario_mix import largest_remainder_counts, scenario_schedule

        self.assertEqual(largest_remainder_counts(200), {
            "buying": 80, "browsing": 80, "intent_override": 30, "boundary": 10,
        })
        self.assertEqual(largest_remainder_counts(250), {
            "buying": 100, "browsing": 100, "intent_override": 38, "boundary": 12,
        })
        schedule = scenario_schedule(240, seed=7)
        self.assertEqual(Counter(schedule), {
            "buying": 96, "browsing": 96, "intent_override": 36, "boundary": 12,
        })
        self.assertEqual(schedule, scenario_schedule(240, seed=7))

    def test_largest_remainder_rejects_negative_counts(self):
        from scripts.scenario_mix import largest_remainder_counts

        with self.assertRaises(ValueError):
            largest_remainder_counts(-1)


class DCPPersistenceTest(unittest.TestCase):
    def test_nonpersistent_stores_never_read_or_write_disk(self):
        import tempfile
        from pathlib import Path
        from src.context_engine import GuidanceLearner, ProfileService

        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profiles.json"
            guidance_path = Path(directory) / "guidance.json"
            profiles = ProfileService(str(profile_path), persistent=False)
            profiles._store["session-only"] = {"prefs": []}
            profiles._flush()
            guidance = GuidanceLearner(str(guidance_path), persistent=False)
            guidance.stats["color"] = 1.0
            guidance._flush()
            self.assertFalse(profile_path.exists())
            self.assertFalse(guidance_path.exists())

    def test_benchmark_agents_get_unique_ephemeral_state_dirs(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.eval_support import new_isolated_agent
        from src.agent import Agent

        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(json.dumps({
                "parent_asin": "A", "title": "boot", "features": [], "details": {},
                "categories": [], "store": "", "description": [], "price": 1.0,
            }) + "\n", encoding="utf-8")
            old_vector = Agent.USE_VECTOR
            Agent.USE_VECTOR = False
            agents = []
            try:
                first = new_isolated_agent(catalog_path)
                second = new_isolated_agent(catalog_path)
                agents.extend((first, second))
                self.assertNotEqual(first._profiles.path.parent, second._profiles.path.parent)
                self.assertFalse(first.USE_DCP)
                self.assertFalse(first._profiles.persistent)
                self.assertEqual(first.POOL_SIZE_OVERRIDE, 200)
            finally:
                for benchmark_agent in agents:
                    benchmark_agent._benchmark_state_dir.cleanup()
                Agent.USE_VECTOR = old_vector


if __name__ == "__main__":
    unittest.main()
