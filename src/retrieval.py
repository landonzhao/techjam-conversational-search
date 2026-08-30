"""Multi-route retrieval: dense (BGE) and Reciprocal Rank Fusion.

BM25 lives in `src/catalog.py` because it is tightly coupled to the FTS5 index.
This module owns the dense track and the fusion step.

To change Buying retrieval: adjust BUYING_VECTOR_WEIGHT in src/config.py.
To change Browsing retrieval: adjust BROWSING_VECTOR_WEIGHT in src/config.py.
To add a new retrieval route: add a function here and fuse it in Agent._retrieve().
To modify hybrid weights: change the `vec_weight` argument to `rrf()`.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import (
    BUYING_VECTOR_WEIGHT, BROWSING_VECTOR_WEIGHT, EMBED_CACHE_ASINS,
    EMBED_CACHE_NPY, EMBED_MODEL, EMBED_QUERY_PREFIX, RRF_K, VECTOR_WEIGHT,
)


# ---------------------------------------------------------------------------
# RRF fusion (standalone — no catalog dependency)

def rrf(
    primary: list[str],
    secondary: list[str],
    secondary_weight: float = VECTOR_WEIGHT,
    k: int = RRF_K,
    top_n: int = 200,
) -> list[str]:
    """Weighted Reciprocal Rank Fusion.

    primary receives weight 1.0; secondary receives `secondary_weight`.
    Returns the top_n ASINs sorted by descending fused score.
    """
    scores: dict[str, float] = {}
    for rank, asin in enumerate(primary):
        scores[asin] = scores.get(asin, 0.0) + 1.0 / (k + rank + 1)
    for rank, asin in enumerate(secondary):
        scores[asin] = scores.get(asin, 0.0) + secondary_weight / (k + rank + 1)
    return sorted(scores, key=lambda a: -scores[a])[:top_n]


# ---------------------------------------------------------------------------
# Calibrated convex-combination fusion (roadmap component I — score-aware rerank fusion)

def _minmax(vals: list[float]) -> list[float]:
    """Min-max normalize to [0,1]; degenerate (all equal) → neutral 0.5 (preserves tie-break)."""
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1e-12:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def convex_fuse(
    pool: list[str],
    primary_scores: dict[str, float],
    secondary_scores: list[float],
    beta: float,
) -> list[str]:
    """Score-aware fusion of two rankers over the head `secondary_scores` covers.

    Unlike RRF (which uses only rank order), this blends the calibrated magnitudes:
        FinalScore(c) = (1 − beta)·minmax(primary)[c] + beta·minmax(secondary)[c]
    computed over the first ``len(secondary_scores)`` candidates (the head the secondary ranker —
    e.g. the cross-encoder — actually scored). The tail beyond that head is kept in the incoming
    (primary) order. Deterministic: ties broken by incoming head index. beta=0 ≈ primary-only,
    beta=1 ≈ secondary-only. Degenerate/empty secondary → incoming order unchanged.

    `primary_scores` is an ASIN→score map (e.g. satisfaction/coverage); `secondary_scores` is a
    positional list aligned to pool[:len]. Returns a full permutation of `pool`.
    """
    if not secondary_scores:
        return list(pool)
    d = len(secondary_scores)
    head = pool[:d]
    p_norm = _minmax([float(primary_scores.get(a, 0.0)) for a in head])
    s_norm = _minmax(list(secondary_scores))
    blended = [(1.0 - beta) * p_norm[i] + beta * s_norm[i] for i in range(d)]
    order = sorted(range(d), key=lambda i: (-blended[i], i))
    return [head[i] for i in order] + list(pool[d:])


# ---------------------------------------------------------------------------
# Dense retrieval

class VectorRetriever:
    """In-memory dense retrieval using precomputed BGE-small-en-v1.5 embeddings.

    Loaded lazily from cache/. If the cache or model is unavailable (offline /
    no cache built) construction raises; Agent falls back to BM25-only.

    To change the embedding model: update EMBED_MODEL in src/config.py and
    rebuild the cache with scripts/build_embeddings.py.
    """

    def __init__(
        self,
        emb_path: str = EMBED_CACHE_NPY,
        asins_path: str = EMBED_CACHE_ASINS,
        model_name: str = EMBED_MODEL,
    ) -> None:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.embeddings = np.load(emb_path)   # (N, 384) float32, L2-normalised
        self.asins: list[str] = json.loads(Path(asins_path).read_text(encoding="utf-8"))
        self.model = SentenceTransformer(model_name)
        self._asin_index: dict[str, int] = {a: i for i, a in enumerate(self.asins)}

    def search(self, query: str, top_n: int) -> list[str]:
        """Encode query, return top_n ASINs by cosine similarity."""
        vec = self.model.encode(
            [EMBED_QUERY_PREFIX + query], normalize_embeddings=True
        )[0].astype("float32")
        return self._top(vec, top_n)

    def search_decayed(self, messages: list[str], top_n: int, decay: float) -> list[str]:
        """Recency-weighted multi-turn dense retrieval.

        Embeds each turn and combines with recency weights so that recent
        constraint messages dominate. decay=1.0 is uniform; <1.0 fades older turns.
        Useful when an Intent Override should let the newest requirement dominate.
        """
        if not messages:
            return []
        vecs = self.model.encode(
            [EMBED_QUERY_PREFIX + m for m in messages], normalize_embeddings=True
        ).astype("float32")
        n = len(messages)
        weights = self.np.array([decay ** (n - 1 - i) for i in range(n)], dtype="float32")
        combined = (weights[:, None] * vecs).sum(axis=0)
        norm = float(self.np.linalg.norm(combined))
        if norm > 0:
            combined /= norm
        return self._top(combined, top_n)

    def phrase_similarities(self, phrases: list[str], asins: list[str]) -> dict[str, float]:
        """Max cosine similarity between any phrase embedding and each candidate's embedding.

        Used by CoverageReranker for semantic (paraphrase-tolerant) constraint scoring.
        Phrases are encoded with the same query prefix used for retrieval.
        """
        if not phrases or not asins:
            return {}
        vecs = self.model.encode(
            [EMBED_QUERY_PREFIX + p for p in phrases], normalize_embeddings=True
        ).astype("float32")  # (P, D)
        result: dict[str, float] = {}
        for asin in asins:
            idx = self._asin_index.get(asin)
            result[asin] = float((vecs @ self.embeddings[idx]).max()) if idx is not None else 0.0
        return result

    def phrase_similarity_matrix(self, phrases: list[str],
                                 asins: list[str]) -> dict[str, list[float]]:
        """Per-phrase cosine similarity to each candidate (NOT max-reduced over phrases).

        Returns {asin: [cos(phrase_0, asin), cos(phrase_1, asin), ...]}. Unlike
        `phrase_similarities` (which max-reduces for a single coverage bonus), the satisfaction
        scorer needs each phrase's similarity separately so it can weight and gate per constraint.
        Cheap: one encode of the few phrases, then a cached-embedding dot product per candidate.
        """
        if not phrases or not asins:
            return {}
        vecs = self.model.encode(
            [EMBED_QUERY_PREFIX + p for p in phrases], normalize_embeddings=True
        ).astype("float32")  # (P, D)
        zeros = [0.0] * len(phrases)
        result: dict[str, list[float]] = {}
        for asin in asins:
            idx = self._asin_index.get(asin)
            result[asin] = (self.embeddings[idx] @ vecs.T).tolist() if idx is not None else list(zeros)
        return result

    def _top(self, vec, top_n: int) -> list[str]:
        scores = self.embeddings @ vec
        n = min(top_n, len(scores))
        idx = self.np.argpartition(scores, -n)[-n:]
        idx = idx[self.np.argsort(scores[idx])[::-1]]
        return [self.asins[i] for i in idx]


# ---------------------------------------------------------------------------
# Intent-aware vector weight (depends only on buying_score and config)

def vector_weight(
    buying_score: float,
    use_intent_routing: bool = True,
    use_confidence_routing: bool = True,
) -> float:
    """Interpolate dense-track weight from the graded buying score.

    buying=1 → BM25-heavy (0.20); browsing=0 → dense-heavy (0.35).
    When intent routing is off, uses the neutral midpoint VECTOR_WEIGHT.
    """
    if not use_intent_routing:
        return VECTOR_WEIGHT
    if use_confidence_routing:
        b = buying_score
        return BROWSING_VECTOR_WEIGHT + b * (BUYING_VECTOR_WEIGHT - BROWSING_VECTOR_WEIGHT)
    if buying_score >= 0.6:
        return BUYING_VECTOR_WEIGHT
    if buying_score <= 0.4:
        return BROWSING_VECTOR_WEIGHT
    return VECTOR_WEIGHT
