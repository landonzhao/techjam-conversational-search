"""Learning-to-Rank feature extraction (docs/ADVANCED_RANKING_PLAN.md, step 1).

Produces a small, leak-robust per-candidate feature vector by reusing the existing ranking signals,
so training (offline) and inference (in the agent) share one code path. Kept deliberately small — few
features generalize better given the distinct public/private users and ~hundreds of training sessions.

A feature that is inherently leak-dependent (verbatim `coverage`) is included as ONE signal among
many; the trained model bounds its weight rather than us trusting it wholesale.
"""
from __future__ import annotations

import json

FEATURE_NAMES = [
    "retrieval_rank",   # normalized position in the fused BM25+dense order (leak-agnostic)
    "satisfaction",     # max(lexical, semantic) constraint-match score
    "coverage",         # verbatim IDF coverage (the leaky signal)
    "cross_encoder",    # ms-marco (query, product) precision score
    "log_popularity",   # log1p(rating_number)
    "avg_rating",       # catalog average rating
    "price_proximity",  # closeness to a disclosed budget (0 when none)
    "category_match",   # title matches the routed category
    "specificity",      # how much the shopper disclosed (context; constant within a session)
]


class RankingFeatures:
    """Extracts per-candidate feature vectors from already-computed ranking signals."""

    def __init__(self, catalog: dict[str, dict], coverage) -> None:
        self.catalog = catalog
        self.cov = coverage           # CoverageReranker: reuse doc/_pop/_price_prox/_cat_match

    @staticmethod
    def _norm_rank(i: int, n: int) -> float:
        return 1.0 - (i / n) if n > 1 else 1.0   # rank 0 -> 1.0 (best), last -> ~0

    def _avg_rating(self, asin: str) -> float:
        try:
            return float(self.catalog.get(asin, {}).get("average_rating") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def extract(
        self,
        asins: list[str],
        *,
        phrases: list[str],
        satisfaction_scores: dict[str, float] | None,
        ce_scores: dict[str, float] | None,
        budget: float | None,
        category: str | None,
    ) -> dict[str, list[float]]:
        """`asins` is the incoming (retrieval-fused) order. Returns {asin: [features]} aligned to
        FEATURE_NAMES. Pure verbatim `coverage` is computed here (IDF-weighted) so it is a signal
        distinct from `satisfaction` (which also uses semantics) — that difference is the leak-vs-
        honest cue LTR learns. Missing score dicts contribute 0 (e.g. cross-encoder off)."""
        n = len(asins)
        sat = satisfaction_scores or {}
        ce = ce_scores or {}
        prepared = self.cov._prepare(phrases)
        spec = min(1.0, len(prepared) / 3.0)
        out: dict[str, list[float]] = {}
        for i, a in enumerate(asins):
            coverage = self.cov._coverage(a, prepared, use_idf=True) if prepared else 0.0
            out[a] = [
                self._norm_rank(i, n),
                float(sat.get(a, 0.0)),
                coverage,
                float(ce.get(a, 0.0)),
                self.cov._pop(a),   # already log1p(rating_number)
                self._avg_rating(a),
                self.cov._price_prox(a, budget) if budget is not None else 0.0,
                float(self.cov._cat_match(a, category)) if category else 0.0,
                spec,
            ]
        return out


class LTRModel:
    """Applies a trained linear LTR model (cache/ltr_model.json) at inference.

    Pure-Python scoring (no numpy) so the agent import stays light. `score_order` re-ranks the
    retrieval-order pool by the learned combination of the RankingFeatures signals.
    """

    def __init__(self, path: str, features: RankingFeatures) -> None:
        d = json.load(open(path, encoding="utf-8"))
        self.w = d["weights"]
        self.b = float(d["intercept"])
        self.mean = d["mean"]
        self.std = [s or 1.0 for s in d["std"]]
        self.features = features

    def score_order(
        self,
        asins: list[str],
        *,
        satisfaction_scores: dict[str, float] | None,
        ce_scores: dict[str, float] | None,
        phrases: list[str],
        budget: float | None,
        category: str | None,
    ) -> tuple[list[str], dict[str, float]]:
        fv = self.features.extract(
            asins, phrases=phrases, satisfaction_scores=satisfaction_scores,
            ce_scores=ce_scores, budget=budget, category=category)
        scores: dict[str, float] = {}
        for a, f in fv.items():
            s = self.b
            for i, x in enumerate(f):
                s += self.w[i] * ((x - self.mean[i]) / self.std[i])
            scores[a] = s
        return sorted(asins, key=lambda a: -scores[a]), scores
