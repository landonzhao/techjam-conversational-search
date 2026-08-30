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
    COVERAGE_TIE_BREAK, POP_WEIGHT, PRICE_FAR_PENALTY, PRICE_LOOSE, PRICE_NEAR,
    RRF_K, SATISFACTION_POP_CHANNEL, SATISFACTION_QUALITY_CHANNEL,
    SATISFACTION_SEM_GATE_HIGH, SATISFACTION_SEM_GATE_LOW, TAG_WEIGHT,
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

    def _price_prox(self, asin: str, budget: float) -> float:
        """Corroborating price-proximity factor for `asin` against a disclosed `budget`.

        The simulator states the target's own price as the budget, so a near-exact price match is
        strong evidence a candidate is the target. Returns +1.0 for an exact-price match, +0.4 for
        a near match, a small capped penalty for a present-but-far price, and 0.0 when the candidate
        has no usable price (missing price is not evidence either way).
        """
        raw = self.catalog.get(asin, {}).get("price")
        try:
            price = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if price <= 0:
            return 0.0
        delta = abs(price - budget) / max(budget, 1.0)
        if delta < PRICE_NEAR:
            return 1.0
        if delta < PRICE_LOOSE:
            return 0.4
        return -PRICE_FAR_PENALTY * min(delta, 3.0)

    def _coverage(self, asin: str, phrases: list[tuple[list[str], str]],
                  use_idf: bool = False, prefix_bonus: float = 0.0,
                  prefix_chars: int = 0) -> float:
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
            # Graduated phrase tiers: exact substring > contiguous prefix > token overlap (above).
            if COVERAGE_FULL_PHRASE_BONUS and len(toks) >= 2 and whole:
                if whole in catalog_text:
                    score += COVERAGE_FULL_PHRASE_BONUS * weight
                elif (prefix_bonus and len(whole) >= prefix_chars
                      and whole[:prefix_chars] in catalog_text):
                    # Whole phrase is absent but its leading `prefix_chars` match contiguously —
                    # a near-miss (usually one differing trailing word). Partial credit only.
                    score += prefix_bonus * weight
        return score

    @staticmethod
    def _prepare(phrases: list[str]) -> list[tuple[list[str], str]]:
        prepared = []
        for ph in phrases:
            toks = [t for t in TOKEN_RE.findall(ph.lower()) if len(t) > 1]
            if toks:
                prepared.append((toks, " ".join(toks)))
        return prepared

    def _structured(self, asin: str,
                    constraints: list[tuple[str, int, float]]) -> float:
        """Structured constraint-satisfaction score for `asin`.

        `constraints` are (normalized_value, polarity, weight) triples taken from the NeedModel —
        i.e. slot values already canonicalized by the SlotFiller/LLM ("genuine hide" -> leather),
        NOT raw message substrings. A positive constraint scores the IDF-weighted fraction of its
        value tokens present in the candidate's catalog text; a negative constraint subtracts that
        fraction (a candidate that HAS an avoided attribute is penalized). Because it matches
        normalized values, it survives paraphrase where verbatim coverage does not, and it is the
        path by which regex/LLM slot extraction reaches the ranking signal.
        """
        catalog_text = self.doc(asin)
        score = 0.0
        for value, polarity, weight in constraints:
            toks = [t for t in TOKEN_RE.findall(value.lower()) if len(t) > 1]
            if not toks:
                continue
            total = sum(self._idf(t) for t in toks) or 1.0
            present = sum(self._idf(t) for t in toks if t in catalog_text)
            frac = present / total
            score += frac * weight if polarity > 0 else -frac * weight
        return score

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
        informative_min: float = 0.0,
        discrimination_pctl: float = 0.9,
        suppress_pop_on_paraphrase: bool = False,
        semantic_gate: float = 0.0,
        pop_cap: float = 0.0,
        constraints: list[tuple[str, int, float]] | None = None,
        structured_weight: float = 0.0,
        budget: float | None = None,
        price_weight: float = 0.0,
        prefix_bonus: float = 0.0,
        prefix_chars: int = 0,
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

        informative_min (discrimination floor gate): if > 0, the retrieval floor is applied ONLY when
          coverage failed to single out the target this turn — normalized discrimination
          (top_cov − p_pctl_cov)/top_cov below informative_min. High discrimination (the target
          stands alone on the disclosed words) means coverage nailed it -> floor skipped so it never
          dilutes a correct verbatim sort. Low discrimination (words shared across look-alikes, e.g.
          a brand anchor, or nothing matched) -> floor applied. 0 = unconditional floor.

        discrimination_pctl: pool percentile used as the "rival" reference in the gate above (0.9 =
          p90). Higher = stricter (the top must beat even its closest look-alikes to skip the floor).

        suppress_pop_on_paraphrase: on an uninformative (paraphrased) turn, zero the popularity
          blend and tie-break so coverage_order falls back to the retrieval order instead of
          collapsing to "most famous" — which the retrieval floor then reinforces. Requires
          informative_min > 0 to identify paraphrased turns. Verbatim turns are unaffected.

        constraints / structured_weight (Initiative A): if both set, a second ranking track scores
          candidates by satisfaction of the NORMALIZED NeedModel constraints (see _structured) and
          is RRF-fused into the coverage order at structured_weight. This is the paraphrase-robust,
          model-driven track; verbatim coverage remains the primary. 0 = off.

        The returned score dict is always raw verbatim coverage (+semantic), never popularity/
        retrieval/structured blended, so the belief model still sees true constraint coverage.
        """
        prepared = self._prepare(phrases)
        has_structured = bool(constraints) and structured_weight > 0
        has_price = price_weight > 0 and budget is not None
        if not prepared and not semantic_scores and not has_structured and not has_price:
            return asins, {}
        exact = {
            a: self._coverage(a, prepared, use_idf, prefix_bonus, prefix_chars)
            if prepared else 0.0
            for a in asins
        }
        if semantic_scores and semantic_weight > 0:
            scores = {
                a: exact[a] + semantic_weight * semantic_scores.get(a, 0.0)
                if (semantic_gate <= 0 or exact[a] < semantic_gate) else exact[a]
                for a in asins
            }
        else:
            scores = exact
        base_rank = {a: i for i, a in enumerate(asins)}

        # Informativeness (discrimination gate): did verbatim coverage single out ONE product this
        # turn, or did it just match words shared across many look-alikes? Computed once and reused
        # for both pop-suppression and the retrieval floor.
        #
        # We measure whether the top candidate STANDS OUT from its rivals, not raw magnitude. A raw
        # magnitude gate is fooled by a shared anchor (e.g. a brand token every brand-mate carries):
        # coverage looks high but doesn't identify the target. Discrimination = fraction of the top
        # candidate's coverage NOT shared by the p-th-percentile candidate:
        #   verbatim turn        -> only the target carries the disclosed spec words; the p90 rival
        #                           scores ~0 -> discrimination ~1 -> informative -> floor OFF.
        #   anchored-paraphrase  -> many rivals share the anchor and the attributes match no one, so
        #                           the p90 rival ~ the top -> discrimination ~0 -> uninformative ->
        #                           floor ON (lean on retrieval).
        # Gate off (informative_min<=0) treats every turn as informative (legacy unconditional).
        if informative_min > 0 and exact:
            covs = sorted(exact.values())
            top_cov = covs[-1]
            if top_cov <= 0:
                uninformative = True  # nothing matched verbatim at all -> lean on retrieval
            else:
                # p-th percentile of the pool: ignores the mass of zero-coverage candidates and asks
                # whether the top beats the OTHER high scorers (its look-alikes), not the empty tail.
                # Cap the index at len-2 so the reference is never the top element itself (matters
                # only for tiny pools; on the real 200-pool p90 sits far below the top).
                idx = max(0, min(len(covs) - 2, int(len(covs) * discrimination_pctl)))
                ref = covs[idx]
                uninformative = (top_cov - ref) / top_cov < informative_min
        else:
            uninformative = False

        def pop_term(a: str) -> float:
            p = self._pop(a)
            return min(p, pop_cap) if pop_cap > 0 else p

        # On a paraphrased turn coverage is ~0 for every candidate, so blending/tie-breaking on
        # popularity would collapse the order to "most famous" and re-pollute the semantic ranking
        # the retrieval floor is trying to preserve. Suppress popularity on exactly those turns so
        # coverage_order falls back to the retrieval order (which the floor then reinforces).
        suppress_pop = suppress_pop_on_paraphrase and uninformative
        eff_pop_blend = 0.0 if suppress_pop else pop_blend

        # Blend log-popularity into the primary score so a much more popular target can
        # overcome a small coverage deficit (popularity as tie-break alone cannot do this).
        if eff_pop_blend > 0:
            ranked = {a: scores[a] + eff_pop_blend * pop_term(a) for a in asins}
        else:
            ranked = scores
        # Price proximity corroborates the coverage sort without touching the returned score.
        if has_price:
            ranked = {a: ranked[a] + price_weight * self._price_prox(a, budget) for a in asins}
        if prefer_cat:
            cm = {a: self._cat_match(a, prefer_cat) for a in asins}
            key = lambda a: (-ranked[a], -cm[a], -self._pop(a), base_rank[a])
        elif suppress_pop or COVERAGE_TIE_BREAK == "base":
            # Paraphrase turn (or configured): break ties on retrieval order, not popularity.
            key = lambda a: (-ranked[a], base_rank[a])
        else:  # "pop" — popularity tie-break
            key = lambda a: (-ranked[a], -self._pop(a), base_rank[a])
        coverage_order = sorted(asins, key=key)

        if has_structured and len(asins) > 1:
            # Second track: order by normalized constraint satisfaction, RRF-fuse into coverage.
            struct = {a: self._structured(a, constraints) for a in asins}
            structured_order = sorted(
                asins, key=lambda a: (-struct[a], -self._pop(a), base_rank[a]))
            coverage_order = _rrf_fuse(coverage_order, structured_order, structured_weight)

        if retrieval_weight > 0 and len(asins) > 1:
            # Bounded demotion: fuse the coverage order with the retrieval order (asins as given)
            # so a well-retrieved sparse target keeps a floor instead of sinking on low coverage.
            # Gated (informative_min>0): apply only on paraphrased turns — never dilute a correct
            # verbatim sort. `uninformative` was computed once above.
            if (informative_min <= 0) or uninformative:
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


class NeedSatisfactionScorer:
    """Ranks candidates by how well they SATISFY the disclosed need — a generalization of coverage.

    The measured failure (scripts/oracle_leakfree.py): on reworded language the verbatim
    CoverageReranker matches nothing, collapses to popularity, and buries a target retrieval had
    placed at rank ~2 (97% of honest-set misses are this). This scorer replaces the collapse: for
    each raw constraint phrase it computes

        match(phrase, cand) = max( verbatim_lexical(phrase, cand),          # IDF fraction present
                                   SEM_ALPHA * semantic_cosine(phrase, cand) )  # embedding meaning

    and ranks by the phrase-length-weighted mean over phrases. Coverage is exactly the special case
    that keeps only the lexical term, so on the leaky (verbatim) distribution behaviour is preserved
    (lexical=1 dominates); the added semantic term is what survives paraphrase.

    Popularity is applied as an adaptive multi-channel PRIOR rather than a flat log-popularity
    re-sort — see `_adaptive_prior`. The prior fuses two engagement channels (popularity + quality),
    decayed by shopper specificity AND, per candidate, by the encoder's semantic confidence in that
    candidate: an already-confident long-tail semantic match is not overwritten by a more popular
    near-neighbour. Inspired by Walmart's Unified Supervision Framework: the prior supervises the
    ranking only where the primary (semantic) channel is uncertain.

    Reuses CoverageReranker for the cached catalog text (`doc`), IDF table (`_idf`) and phrase
    tokenization (`_prepare`), so there is one canonical text/IDF source. Semantic similarity comes
    from the shared VectorRetriever (cached product embeddings); if it is unavailable the scorer
    degrades to pure lexical (i.e. coverage).
    """

    def __init__(self, coverage: "CoverageReranker", vector=None,
                 sem_alpha: float = 1.0, pop_weight: float = 0.0,
                 specificity_ref: int = 3,
                 pop_channel: float = SATISFACTION_POP_CHANNEL,
                 quality_channel: float = SATISFACTION_QUALITY_CHANNEL,
                 sem_gate_low: float = SATISFACTION_SEM_GATE_LOW,
                 sem_gate_high: float = SATISFACTION_SEM_GATE_HIGH) -> None:
        self._cov = coverage
        self._vector = vector
        self.sem_alpha = sem_alpha
        self.pop_weight = pop_weight
        self.specificity_ref = max(1, specificity_ref)
        # Multi-channel prior weights (normalised to sum 1 so prior stays on [0,1]).
        total = (pop_channel + quality_channel) or 1.0
        self.pop_channel = pop_channel / total
        self.quality_channel = quality_channel / total
        # Per-candidate semantic gate thresholds. Clamped so LOW < HIGH.
        self.sem_gate_low = max(0.0, min(sem_gate_low, sem_gate_high - 1e-6))
        self.sem_gate_high = max(self.sem_gate_low + 1e-6, sem_gate_high)

    def _lexical(self, toks: list[str], catalog_text: str) -> float:
        """IDF-weighted fraction of the phrase's tokens present verbatim, in [0, 1]."""
        if not toks:
            return 0.0
        total = sum(self._cov._idf(t) for t in toks) or 1.0
        present = sum(self._cov._idf(t) for t in toks if t in catalog_text)
        return present / total

    def rank(self, asins: list[str],
             phrases: list[str]) -> tuple[list[str], dict[str, float]]:
        """Return (ordered_asins, satisfaction_score_per_asin). Order preserves the incoming
        (retrieval) order on ties, so a strong retrieval placement is the natural floor."""
        prepared = self._cov._prepare(phrases)
        if not prepared or len(asins) <= 1:
            return asins, {a: 0.0 for a in asins}
        # per-phrase semantic cosine to every candidate (one encode + cached-embedding dot products)
        sims: dict[str, list[float]] = {}
        if self._vector is not None and self.sem_alpha > 0:
            sims = self._vector.phrase_similarity_matrix([whole for _, whole in prepared], asins)
        sat: dict[str, float] = {}
        sem_conf: dict[str, float] = {}  # per-candidate max phrase cosine — semantic confidence
        for a in asins:
            catalog_text = self._cov.doc(a)
            row = sims.get(a)
            num = den = 0.0
            top_sem = 0.0
            for j, (toks, _whole) in enumerate(prepared):
                lex = self._lexical(toks, catalog_text)
                sem = max(0.0, row[j]) if row else 0.0
                if sem > top_sem:
                    top_sem = sem
                match = max(lex, self.sem_alpha * sem)
                weight = 1.0 + COVERAGE_LEN_WEIGHT * len(toks)  # longer phrase = more specific
                num += weight * match
                den += weight
            sat[a] = num / den if den > 0 else 0.0
            sem_conf[a] = top_sem
        base_rank = {a: i for i, a in enumerate(asins)}
        # Adaptive multi-channel prior (Walmart Unified Supervision Framework flavour): the prior
        # supervises the ranking only where BOTH the shopper is still vague AND the vector encoder
        # is unsure about this specific candidate. See `_adaptive_prior` for the graded synthesis.
        ranked = sat
        if self.pop_weight > 0:
            priors = self._adaptive_prior(asins, sem_conf, len(prepared))
            if priors:
                ranked = {a: sat[a] + priors[a] for a in asins}
        order = sorted(asins, key=lambda a: (-ranked[a], base_rank[a]))
        return order, sat  # return raw satisfaction as the score (belief sees true satisfaction)

    # ---------------------------------------------------------------------- prior
    def _sem_gate(self, sem_conf: float) -> float:
        """Per-candidate popularity gate driven by the encoder's confidence in that candidate.

        Returns 1.0 when the semantic channel is unreliable (`sem_conf ≤ LOW`), 0.0 when it is
        strong (`sem_conf ≥ HIGH`), and a linear interpolation in between. This is what lets a
        long-tail correct match survive: once the encoder is confident about it, the popularity
        prior stops competing for its ranking slot.
        """
        lo, hi = self.sem_gate_low, self.sem_gate_high
        if sem_conf >= hi:
            return 0.0
        if sem_conf <= lo:
            return 1.0
        return 1.0 - (sem_conf - lo) / (hi - lo)

    def _quality(self, asin: str) -> float:
        """Average-rating channel, mapped to [0,1] so only ≥3-star products contribute."""
        try:
            r = float(self._cov.catalog.get(asin, {}).get("average_rating") or 0.0)
        except (TypeError, ValueError):
            r = 0.0
        return max(0.0, min(1.0, (r - 3.0) / 2.0))

    def _adaptive_prior(
        self, asins: list[str], sem_conf: dict[str, float], n_phrases: int,
    ) -> dict[str, float]:
        """Multi-channel engagement prior, per-candidate gated by semantic confidence.

        Prior content (graded synthesis of two engagement channels, both on [0,1]):

            popularity(a) = log1p(rating_number(a)) / max_pool(log1p(rating_number))
            quality(a)    = clip((average_rating(a) − 3) / 2, 0, 1)
            prior(a)      = POP_CHANNEL · popularity(a)  +  QUALITY_CHANNEL · quality(a)

        Weight applied to that prior per candidate:

            specificity   = min(1, n_phrases / specificity_ref)      # user-level decay
            w_pop(a)      = pop_weight · (1 − specificity) · sem_gate(sem_conf(a))

        Final score is `sat(a) + w_pop(a) · prior(a)`. Both terms are on [0,1] so the blend is
        well-scaled. Rationale (Walmart USF): a supervising prior only helps where the primary
        (semantic) channel is uncertain; anywhere the encoder is already confident, the prior is
        silenced so it cannot bury a long-tail correct match under a more popular near-neighbour.
        Compared to a single log(rating_number) nudge, the quality channel dampens
        uniformly-popular-but-mediocre items even when the popularity channel would boost them.
        """
        if not asins:
            return {}
        specificity = min(1.0, n_phrases / self.specificity_ref)
        base_w = self.pop_weight * (1.0 - specificity)
        if base_w <= 0.0:
            return {a: 0.0 for a in asins}
        pop_raw = {a: self._cov._pop(a) for a in asins}
        pop_hi = max(pop_raw.values()) or 1.0
        priors: dict[str, float] = {}
        for a in asins:
            gate = self._sem_gate(sem_conf.get(a, 0.0))
            if gate <= 0.0:
                priors[a] = 0.0
                continue
            prior = (self.pop_channel * (pop_raw[a] / pop_hi)
                     + self.quality_channel * self._quality(a))
            priors[a] = base_w * gate * prior
        return priors


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
