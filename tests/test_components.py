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


class ConvexFuseTest(unittest.TestCase):
    def test_ce_magnitude_reorders_head(self):
        # primary (satisfaction) prefers A; CE strongly prefers C. At beta=1 (CE only) C leads.
        from src.retrieval import convex_fuse
        pool = ["A", "B", "C"]
        sat = {"A": 1.0, "B": 0.5, "C": 0.0}
        ce = [0.1, 0.2, 0.9]  # aligned to pool[:3]
        self.assertEqual(convex_fuse(pool, sat, ce, beta=1.0)[0], "C")
        # beta=0 (primary only) keeps A on top despite CE
        self.assertEqual(convex_fuse(pool, sat, ce, beta=0.0)[0], "A")

    def test_blend_middle(self):
        # A: high sat/low ce; C: low sat/high ce. beta=0.5 → blended tie broken by head index → A.
        from src.retrieval import convex_fuse
        order = convex_fuse(["A", "B", "C"], {"A": 1.0, "B": 0.0, "C": 0.0}, [0.0, 0.0, 1.0], 0.5)
        self.assertEqual(set(order), {"A", "B", "C"})  # permutation preserved
        self.assertEqual(order[0], "A")

    def test_empty_ce_preserves_order(self):
        from src.retrieval import convex_fuse
        self.assertEqual(convex_fuse(["A", "B"], {"A": 1.0}, [], 0.6), ["A", "B"])

    def test_degenerate_scores_preserve_order(self):
        # all-equal primary and secondary → neutral 0.5 → stable tie-break keeps incoming order
        from src.retrieval import convex_fuse
        self.assertEqual(
            convex_fuse(["A", "B", "C"], {"A": 1.0, "B": 1.0, "C": 1.0}, [3.0, 3.0, 3.0], 0.6),
            ["A", "B", "C"])

    def test_tail_beyond_ce_head_preserved(self):
        # CE only scored the first 2; the tail keeps incoming order and is appended after the head.
        from src.retrieval import convex_fuse
        order = convex_fuse(["A", "B", "C", "D"], {"A": 0.0, "B": 1.0}, [0.9, 0.1], beta=1.0)
        self.assertEqual(order[0], "A")       # CE prefers A within the head
        self.assertEqual(order[2:], ["C", "D"])  # tail untouched


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
class NeedModelReviseTest(unittest.TestCase):
    def _c(self, slot, value, pol=1, turn=0):
        from src.understanding import Constraint
        return Constraint(slot=slot, value=value, polarity=pol, turn=turn)

    def test_single_valued_category_overwrites(self):
        # category is single-valued: 'sandal' (turn 3) supersedes 'boot' (turn 1)
        from src.understanding import NeedModel
        n = NeedModel()
        n.revise([self._c("category", "boot", turn=1)])
        n.revise([self._c("category", "sandal", turn=3)])
        cats = [c.value for c in n.positives("category")]
        self.assertEqual(cats, ["sandal"])
        self.assertEqual(n.category, "sandal")

    def test_multi_valued_color_coexists(self):
        # color is multi-valued: 'black or navy' both survive
        from src.understanding import NeedModel
        n = NeedModel()
        n.revise([self._c("color", "black", turn=1)])
        n.revise([self._c("color", "navy", turn=1)])
        self.assertEqual({c.value for c in n.positives("color")}, {"black", "navy"})

    def test_single_valued_size_overwrites(self):
        from src.understanding import NeedModel
        n = NeedModel()
        n.revise([self._c("size", "m", turn=1)])
        n.revise([self._c("size", "l", turn=2)])
        self.assertEqual([c.value for c in n.positives("size")], ["l"])

    def test_negative_does_not_trigger_overwrite(self):
        # a negative constraint on a single-valued slot must not wipe a positive value
        from src.understanding import NeedModel
        n = NeedModel()
        n.revise([self._c("category", "boot", turn=1)])
        n.revise([self._c("category", "sandal", pol=-1, turn=2)])
        self.assertIn("boot", [c.value for c in n.positives("category")])

    # Rule (a): same-turn negation wins
    def test_same_turn_negation_drops_positive(self):
        # "polyester instead of linen": negation and positive arrive in same batch → linen dropped
        from src.understanding import NeedModel
        n = NeedModel()
        n.revise([
            self._c("material", "linen", pol=1, turn=2),
            self._c("material", "linen", pol=-1, turn=2),
        ])
        self.assertEqual([c.value for c in n.positives("material")], [])
        self.assertEqual(len(n.negatives("material")), 1)

    def test_same_turn_negation_other_positive_survives(self):
        # "polyester instead of linen" — polyester positive must survive
        from src.understanding import NeedModel
        n = NeedModel()
        n.revise([
            self._c("material", "linen", pol=1, turn=2),
            self._c("material", "linen", pol=-1, turn=2),
            self._c("material", "polyester", pol=1, turn=2),
        ])
        self.assertIn("polyester", [c.value for c in n.positives("material")])
        self.assertNotIn("linen", [c.value for c in n.positives("material")])

    # Rule (b): category-switch retires stale prior-turn modifiers — only on override turns
    def test_category_switch_clears_prior_modifiers_on_override(self):
        from src.understanding import NeedModel
        n = NeedModel()
        n.revise([self._c("category", "boot", turn=1)])
        n.revise([self._c("color", "brown", turn=1)])
        n.revise([self._c("category", "sandal", turn=3)], is_override=True)
        colors = [c.value for c in n.positives("color")]
        self.assertEqual(colors, [], f"stale 'brown' should clear on override category switch, got {colors}")
        self.assertEqual(n.category, "sandal")

    def test_category_switch_does_not_clear_on_normal_turn(self):
        # Rule (b) must NOT fire on a normal turn — only on is_override=True
        from src.understanding import NeedModel
        n = NeedModel()
        n.revise([self._c("category", "boot", turn=1)])
        n.revise([self._c("color", "brown", turn=1)])
        n.revise([self._c("category", "sandal", turn=3)], is_override=False)
        colors = [c.value for c in n.positives("color")]
        self.assertIn("brown", colors, "non-override category change must NOT clear prior color")

    def test_category_switch_preserves_current_turn_modifiers(self):
        from src.understanding import NeedModel
        n = NeedModel()
        n.revise([self._c("category", "boot", turn=1)])
        n.revise([self._c("color", "brown", turn=1)])
        n.revise([
            self._c("category", "sandal", turn=3),
            self._c("color", "black", turn=3),
        ], is_override=True)
        colors = [c.value for c in n.positives("color")]
        self.assertIn("black", colors)
        self.assertNotIn("brown", colors)


class CategoryGateTest(unittest.TestCase):
    def _cat(self):
        return {
            "SANDAL": {"title": "Leevar Square Toe Heeled Sandals for Women"},
            "BOOT": {"title": "TOMS Women's Chelsea Boots"},
            "AMBIG": {"title": "Comfortable footwear for everyday"},  # no category noun
        }

    def test_wrong_category_demoted(self):
        from src.understanding import apply_category_gate
        out = apply_category_gate(["BOOT", "SANDAL"], "sandal", self._cat())
        self.assertEqual(out, ["SANDAL", "BOOT"])  # boot demoted below the sandal

    def test_unknown_category_not_demoted(self):
        # a candidate whose title resolves to no category is left in place (never demoted on a guess)
        from src.understanding import apply_category_gate
        out = apply_category_gate(["AMBIG", "BOOT"], "sandal", self._cat())
        self.assertEqual(out[0], "AMBIG")

    def test_no_need_category_is_noop(self):
        from src.understanding import apply_category_gate
        self.assertEqual(apply_category_gate(["BOOT", "SANDAL"], None, self._cat()),
                         ["BOOT", "SANDAL"])


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
        # sem_alpha=0 disables the semantic term -> a stub vector cannot change the order.
        # With no lexical overlap, all candidates are catalog-silent and receive unknown_floor.
        # Input order is preserved (tied scores fall back to retrieval rank).
        class StubVector:
            def phrase_similarity_matrix(self, phrases, asins):
                return {a: [0.9] for a in asins}
        from src.ranking import CoverageReranker, NeedSatisfactionScorer
        cat = {
            "T": {"title": "corduroy jacket", "features": "napped pile warm"},
            "P": {"title": "plain cotton tee", "features": "basic"},
            "Q": {"title": "leather belt", "features": "buckle"},
        }
        scorer = NeedSatisfactionScorer(CoverageReranker(cat), vector=StubVector(),
                                        sem_alpha=0.0, unknown_floor=0.0)
        order, sat = scorer.rank(["P", "Q", "T"], ["ribbed velvety fabric"])
        self.assertEqual(order, ["P", "Q", "T"])
        self.assertEqual(max(sat.values()), 0.0)

    def test_unknown_floor_applied_to_silent_candidates(self):
        # With no matching phrases, every candidate is catalog-silent.
        # unknown_floor=0.5 should give each candidate score 0.5, not 0.
        from src.ranking import CoverageReranker, NeedSatisfactionScorer
        cat = {
            "A": {"title": "blue jacket", "features": "warm"},
            "B": {"title": "red hat", "features": "casual"},
        }
        scorer = NeedSatisfactionScorer(CoverageReranker(cat), unknown_floor=0.5)
        order, sat = scorer.rank(["A", "B"], ["completely unrelated xyz phrase"])
        self.assertAlmostEqual(sat["A"], 0.5)
        self.assertAlmostEqual(sat["B"], 0.5)
        # Input order preserved when scores are tied.
        self.assertEqual(order, ["A", "B"])

    def test_unknown_floor_not_applied_when_evidence_exists(self):
        # A candidate WITH a lexical match must NOT be floored to 0.5.
        from src.ranking import CoverageReranker, NeedSatisfactionScorer
        cat = {
            "A": {"title": "blue jacket", "features": "warm"},
            "B": {"title": "completely different", "features": "other stuff"},
        }
        scorer = NeedSatisfactionScorer(CoverageReranker(cat), unknown_floor=0.5)
        _order, sat = scorer.rank(["A", "B"], ["blue jacket"])
        self.assertGreater(sat["A"], 0.5)  # A has evidence; must be above floor

    def test_unknown_floor_zero_preserves_old_behaviour(self):
        # unknown_floor=0.0 keeps the old behaviour: silent candidates score 0.
        from src.ranking import CoverageReranker, NeedSatisfactionScorer
        cat = {"A": {"title": "blue jacket"}, "B": {"title": "red hat"}}
        scorer = NeedSatisfactionScorer(CoverageReranker(cat), unknown_floor=0.0)
        _order, sat = scorer.rank(["A", "B"], ["completely unrelated xyz phrase"])
        self.assertAlmostEqual(sat["A"], 0.0)
        self.assertAlmostEqual(sat["B"], 0.0)


class RetrievalGuardTest(unittest.TestCase):
    """guard_retrieval_head: force-keep retrieval top-K in visible window."""

    def _guard(self, retrieval, ranked, k=3, visible=5):
        from src.ranking import guard_retrieval_head
        return guard_retrieval_head(retrieval, ranked, k, visible)

    def test_retrieval_top_preserved_when_ranked_out(self):
        # Retrieval placed A at rank 0, ranker pushed it to rank 5 (outside visible 5).
        retrieval = ["A", "B", "C", "D", "E", "F"]
        ranked    = ["X", "Y", "Z", "W", "V", "A", "B", "C", "D", "E"]
        order, inserted = self._guard(retrieval, ranked)
        self.assertIn("A", order[:5])
        self.assertIn("A", inserted)

    def test_already_present_not_duplicated(self):
        retrieval = ["A", "B", "C"]
        ranked    = ["A", "B", "C", "D", "E"]
        order, inserted = self._guard(retrieval, ranked)
        self.assertEqual(inserted, [])
        self.assertEqual(order[:3], ["A", "B", "C"])

    def test_order_within_window_preserves_ranked_order(self):
        # After guard inserts A back in, the window must stay in ranked_order score order.
        retrieval = ["A", "B", "C", "D", "E"]
        ranked    = ["C", "D", "E", "F", "G", "A", "B"]
        order, inserted = self._guard(retrieval, ranked, k=2, visible=5)
        # All inserted must appear within visible_k=5.
        for a in inserted:
            self.assertIn(a, order[:5])
        # Score order preserved: C before D before E within visible.
        visible = order[:5]
        if "C" in visible and "D" in visible:
            self.assertLess(visible.index("C"), visible.index("D"))

    def test_guard_k_zero_is_noop(self):
        retrieval = ["A", "B", "C"]
        ranked    = ["X", "Y", "Z", "A", "B"]
        order, inserted = self._guard(retrieval, ranked, k=0)
        self.assertEqual(order, ranked)
        self.assertEqual(inserted, [])


class AdaptivePriorTest(unittest.TestCase):
    """Merged teammate multi-channel prior + per-candidate semantic gate (branch-ranking)."""
    def _scorer(self):
        from src.ranking import CoverageReranker, NeedSatisfactionScorer
        cat = {
            "POP": {"title": "popular tee", "rating_number": 100000, "average_rating": 4.8},
            "TAIL": {"title": "rare tee", "rating_number": 5, "average_rating": 4.9},
        }
        return NeedSatisfactionScorer(CoverageReranker(cat), vector=None, pop_weight=0.3)

    def test_sem_gate_curve(self):
        s = self._scorer()
        self.assertAlmostEqual(s._sem_gate(0.0), 1.0)          # unreliable → full prior
        self.assertAlmostEqual(s._sem_gate(0.9), 0.0)          # confident → silenced
        self.assertTrue(0.0 < s._sem_gate(0.45) < 1.0)         # linear between

    def test_prior_silenced_for_semantically_confident_candidate(self):
        s = self._scorer()
        pri = s._adaptive_prior(["POP", "TAIL"], {"POP": 0.0, "TAIL": 0.9}, n_phrases=1)
        self.assertEqual(pri["TAIL"], 0.0)          # long-tail confident match protected
        self.assertGreater(pri["POP"], 0.0)         # uncertain popular item gets the nudge

    def test_pop_weight_override_zero_disables_prior(self):
        # my NL-turn path passes pop_weight=0 → prior fully off (ties fall back to retrieval order)
        s = self._scorer()
        pri = s._adaptive_prior(["POP", "TAIL"], {"POP": 0.0, "TAIL": 0.0}, n_phrases=1, pop_weight=0.0)
        self.assertEqual(set(pri.values()), {0.0})


if __name__ == "__main__":
    unittest.main()
