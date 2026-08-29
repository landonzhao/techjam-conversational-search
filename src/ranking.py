"""Reranking: Personalizer (popularity + profile tags) then CoverageReranker (verbatim constraints).

Coverage must run last — it resolves verbatim constraint phrases that single out the exact target.
Optional CrossEncoder/LLM rerankers live in src/reranker.py and are fused via RRF before coverage.
"""
from __future__ import annotations

import math
import re

from src.catalog import TOKEN_RE, text, terms
from src.config import (
    COVERAGE_FULL_PHRASE_BONUS, COVERAGE_LEN_WEIGHT, COVERAGE_POP_BLEND,
    COVERAGE_TIE_BREAK, POP_WEIGHT, RRF_K, TAG_WEIGHT,
)


def _rrf_fuse(primary: list[str], secondary: list[str], secondary_weight: float,
              k: int = RRF_K) -> list[str]:
    """Reciprocal-rank fuse two orderings of the same items (primary weight 1.0)."""
    score: dict[str, float] = {}
    for rank, a in enumerate(primary):
        score[a] = score.get(a, 0.0) + 1.0 / (k + rank + 1)
    for rank, a in enumerate(secondary):
        score[a] = score.get(a, 0.0) + secondary_weight / (k + rank + 1)
    return sorted(score, key=lambda a: -score[a])


# ---------------------------------------------------------------------------
class Personalizer:
    """Blends popularity prior and profile-tag overlap into the retrieval rank.

    log(rating_number) is the single largest ranking improvement (MRR 0.565→0.66);
    tag overlap provides additional soft personalization from the anonymized profile.
    """

    def __init__(self, catalog: dict[str, dict]) -> None:
        self.catalog = catalog

    def _pop(self, asin: str) -> float:
        try:
            return math.log1p(float(self.catalog.get(asin, {}).get("rating_number") or 0))
        except (TypeError, ValueError):
            return 0.0

    def _profile_terms(self, profile: dict) -> set[str]:
        tags = profile.get("preference_tags") or []
        summary = profile.get("summary") or ""
        return set(terms(" ".join(tags) + " " + summary))

    def rerank(self, asins: list[str], profile: dict, strength: float) -> list[str]:
        """Rerank by (incoming_rank − popularity_boost − tag_boost), ascending.

        `strength` scales the tag component (0.25 buying, 0.5 browsing).
        """
        pterms = self._profile_terms(profile)
        scored: list[tuple[float, int, str]] = []
        for rank, asin in enumerate(asins):
            product = self.catalog.get(asin, {})
            blob = terms(
                text(product.get("title")) + " "
                + text(product.get("features")) + " "
                + text(product.get("categories"))
            )
            overlap = len(pterms.intersection(blob))
            boost = POP_WEIGHT * self._pop(asin) + strength * TAG_WEIGHT * overlap
            scored.append((rank - boost, rank, asin))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [asin for _, _, asin in scored]


# ---------------------------------------------------------------------------
class CoverageReranker:
    """Scores candidates by verbatim coverage of disclosed constraint phrases.

    The simulator surfaces constraints as verbatim substrings of the target product's
    features/details. Token-level coverage singles out the exact ASIN among near-duplicates
    that BM25/dense/popularity cannot resolve (pool hit-proxy 0.870→0.965, MRR 0.73→0.86).

    Must run last in the reranking stack; tie-break falls back to popularity.
    """

    def __init__(self, catalog: dict[str, dict]) -> None:
        self.catalog = catalog
        self._text_cache: dict[str, str] = {}
        self._n_docs = max(1, len(catalog))
        self._df: dict[str, int] | None = None  # token → document frequency (lazy)

    def _idf(self, token: str) -> float:
        """Inverse document frequency: rare tokens are more discriminative.

        A token in few catalog products (e.g. a distinctive feature word) singles out
        the target; a token in many products (e.g. "cotton") barely narrows anything.
        The df table is built once on first use from the full catalog.
        """
        if self._df is None:
            df: dict[str, int] = {}
            for asin in self.catalog:
                for tok in set(TOKEN_RE.findall(self.doc(asin))):
                    df[tok] = df.get(tok, 0) + 1
            self._df = df
        df_t = self._df.get(token, 0)
        return math.log(self._n_docs / (1 + df_t)) + 1.0

    def doc(self, asin: str) -> str:
        """Concatenated, lowercased catalog text for `asin` (cached)."""
        cached = self._text_cache.get(asin)
        if cached is not None:
            return cached
        p = self.catalog.get(asin, {})
        blob = " ".join((
            text(p.get("title")), text(p.get("features")), text(p.get("details")),
            text(p.get("categories")), text(p.get("store")), text(p.get("description")),
        )).lower()
        blob = re.sub(r"\s+", " ", blob)
        self._text_cache[asin] = blob
        return blob

    def _pop(self, asin: str) -> float:
        try:
            return math.log1p(float(self.catalog.get(asin, {}).get("rating_number") or 0))
        except (TypeError, ValueError):
            return 0.0

    def _coverage(self, asin: str, phrases: list[tuple[list[str], str]],
                  use_idf: bool = False) -> float:
        catalog_text = self.doc(asin)
        score = 0.0
        for toks, whole in phrases:
            weight = 1.0 + COVERAGE_LEN_WEIGHT * len(toks)
            if use_idf:
                # Weight each token by rarity so covering a distinctive token outscores
                # covering a common one (which every lookalike also covers).
                total = sum(self._idf(t) for t in toks) or 1.0
                present = sum(self._idf(t) for t in toks if t in catalog_text)
                score += (present / total) * weight
            else:
                present = sum(1 for t in toks if t in catalog_text)
                score += (present / len(toks)) * weight
            if COVERAGE_FULL_PHRASE_BONUS and len(toks) >= 2 and whole and whole in catalog_text:
                score += COVERAGE_FULL_PHRASE_BONUS * weight
        return score

    @staticmethod
    def _prepare(phrases: list[str]) -> list[tuple[list[str], str]]:
        prepared = []
        for ph in phrases:
            toks = [t for t in TOKEN_RE.findall(ph.lower()) if len(t) > 1]
            if toks:
                prepared.append((toks, " ".join(toks)))
        return prepared

    def rerank_scored(
        self,
        asins: list[str],
        phrases: list[str],
        prefer_cat: str | None = None,
        semantic_scores: dict[str, float] | None = None,
        semantic_weight: float = 0.0,
        use_idf: bool = False,
        pop_blend: float = 0.0,
        retrieval_weight: float = 0.0,
        semantic_gate: float = 0.0,
        pop_cap: float = 0.0,
    ) -> tuple[list[str], dict[str, float]]:
        """Rerank and return (ordered_list, coverage_score_per_asin).

        prefer_cat (measured −0.019; available for ablation): on a tie, prefer the candidate
        whose title matches the shopper's category before falling back to popularity.

        semantic_scores / semantic_weight: cosine similarity added on top of exact coverage.
        semantic_gate (Fix 2): if > 0, add semantic only to candidates whose exact coverage is
          below this threshold — a rescue signal for sparsely-described items where lexical
          coverage fails, without disturbing items lexical coverage already resolves.

        use_idf: weight coverage tokens by rarity.

        pop_blend / pop_cap (Fix 3): blend log-popularity into the score. pop_cap > 0 caps the
          popularity term so ultra-popular lookalikes cannot bury a low-popularity target.

        retrieval_weight (Fix 1, bounded demotion): if > 0, the final order is an RRF fusion of
          the coverage ranking with the incoming retrieval ranking. This stops coverage from
          sinking a strongly-retrieved but sparsely-described target out of the top-k — coverage
          sharpens the order, retrieval provides a floor. 0 reproduces the pure-coverage sort.

        The returned score dict is always raw coverage (+semantic), never popularity/retrieval
        blended, so the belief model still sees true constraint coverage.
        """
        prepared = self._prepare(phrases)
        if not prepared and not semantic_scores:
            return asins, {}
        exact = {a: self._coverage(a, prepared, use_idf) if prepared else 0.0 for a in asins}
        if semantic_scores and semantic_weight > 0:
            scores = {
                a: exact[a] + semantic_weight * semantic_scores.get(a, 0.0)
                if (semantic_gate <= 0 or exact[a] < semantic_gate) else exact[a]
                for a in asins
            }
        else:
            scores = exact
        base_rank = {a: i for i, a in enumerate(asins)}

        def pop_term(a: str) -> float:
            p = self._pop(a)
            return min(p, pop_cap) if pop_cap > 0 else p

        # Blend log-popularity into the primary score so a much more popular target can
        # overcome a small coverage deficit (popularity as tie-break alone cannot do this).
        if pop_blend > 0:
            ranked = {a: scores[a] + pop_blend * pop_term(a) for a in asins}
        else:
            ranked = scores
        if prefer_cat:
            cm = {a: self._cat_match(a, prefer_cat) for a in asins}
            key = lambda a: (-ranked[a], -cm[a], -self._pop(a), base_rank[a])
        elif COVERAGE_TIE_BREAK == "base":
            key = lambda a: (-ranked[a], base_rank[a])
        else:  # "pop" — popularity tie-break
            key = lambda a: (-ranked[a], -self._pop(a), base_rank[a])
        coverage_order = sorted(asins, key=key)

        if retrieval_weight > 0 and len(asins) > 1:
            # Bounded demotion: fuse the coverage order with the retrieval order (asins as given)
            # so a well-retrieved sparse target keeps a floor instead of sinking on low coverage.
            fused = _rrf_fuse(coverage_order, asins, retrieval_weight)
            return fused, scores
        return coverage_order, scores

    def rerank(self, asins: list[str], phrases: list[str],
               prefer_cat: str | None = None) -> list[str]:
        return self.rerank_scored(asins, phrases, prefer_cat)[0]

    def _cat_match(self, asin: str, cat: str) -> int:
        # title only — the categories field is polluted by Amazon's top-level path
        title = text(self.catalog.get(asin, {}).get("title")).lower()
        return 1 if re.search(rf"\b{re.escape(cat)}s?\b", title) else 0


class Diversifier:
    """Maximal Marginal Relevance re-ordering to avoid a wall of near-identical items.

    Keeps a protected head (the confident top picks — usually where the target sits)
    exactly as ranked, then fills the remaining slots by trading relevance against
    novelty: each next pick is penalised for looking like items already chosen. This
    surfaces distinct styles/features instead of ten popular lookalikes, which matters
    for fashion browsing without disturbing the strong top of the list.
    """

    _WORD_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, catalog: dict[str, dict]) -> None:
        self.catalog = catalog
        self._sig_cache: dict[str, frozenset[str]] = {}

    def _signature(self, asin: str) -> frozenset[str]:
        """Distinctive title tokens for similarity comparison (title carries the 'look')."""
        cached = self._sig_cache.get(asin)
        if cached is not None:
            return cached
        title = text(self.catalog.get(asin, {}).get("title")).lower()
        sig = frozenset(t for t in self._WORD_RE.findall(title) if len(t) > 2)
        self._sig_cache[asin] = sig
        return sig

    @staticmethod
    def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / (len(a) + len(b) - inter)

    def reorder(
        self, order: list[str], scores: dict[str, float],
        head_keep: int, top_k: int, lam: float,
    ) -> list[str]:
        """Return `order` with positions [head_keep:top_k] MMR-diversified.

        head_keep: leading positions left untouched (protects the likely target).
        lam: 1.0 = pure relevance (no diversification); 0.0 = pure novelty.
        """
        if len(order) <= head_keep + 1 or lam >= 1.0:
            return order
        selected = list(order[:head_keep])
        pool = list(order[head_keep:])
        sel_sigs = [self._signature(a) for a in selected]
        # normalise relevance to [0,1] over the pool so lam trades on a comparable scale
        pool_scores = [scores.get(a, 0.0) for a in pool]
        lo, hi = min(pool_scores), max(pool_scores)
        span = (hi - lo) or 1.0
        while pool and len(selected) < top_k:
            best_i, best_val = 0, None
            for i, a in enumerate(pool):
                rel = (scores.get(a, 0.0) - lo) / span
                sig = self._signature(a)
                novelty = 1.0 - max((self._jaccard(sig, s) for s in sel_sigs), default=0.0)
                val = lam * rel + (1.0 - lam) * novelty
                if best_val is None or val > best_val:
                    best_i, best_val = i, val
            pick = pool.pop(best_i)
            selected.append(pick)
            sel_sigs.append(self._signature(pick))
        return selected + pool
