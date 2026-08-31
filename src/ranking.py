"""Reranking: Personalizer (popularity + profile tags) then ledger-aware coverage fusion.

Coverage runs last — it combines exact catalog evidence with retrieval and satisfaction signals.
Optional CrossEncoder/LLM rerankers live in src/reranker.py and are fused via RRF before coverage.
"""
from __future__ import annotations

import math
import re

from src.catalog import STOPWORDS, TOKEN_RE, text, terms
from src.config import (
    COVERAGE_FULL_PHRASE_BONUS, COVERAGE_LEN_WEIGHT, COVERAGE_POP_BLEND,
    COVERAGE_TIE_BREAK, POP_WEIGHT, PRICE_FAR_PENALTY, PRICE_LOOSE, PRICE_NEAR,
    RRF_K, TAG_WEIGHT,
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

    def exact_scores(self, asins: list[str], phrases: list[str],
                     use_idf: bool = False) -> dict[str, float]:
        """Return positive lexical coverage evidence without imposing an ordering.

        The dual-track scorer needs the raw evidence and its discrimination statistics separately
        from the final sort.  Missing catalog text contributes zero evidence, never a negative
        score; retrieval/satisfaction terms remain responsible for sparse listings.
        """
        prepared = self._prepare(phrases)
        if not prepared:
            return {asin: 0.0 for asin in asins}
        return {
            asin: self._coverage(asin, prepared, use_idf=use_idf)
            for asin in asins
        }

    def exact_match_counts(self, asins: list[str], phrases: list[str]) -> dict[str, int]:
        """Count complete exact-string phrase matches for leak discrimination."""
        prepared = self._prepare(phrases)
        counts: dict[str, int] = {}
        for asin in asins:
            # Canonicalise punctuation so catalog fields such as ``Material:alloy`` still count as
            # the same exact shopper phrase ``Material alloy``. This is still lexical exactness,
            # not semantic similarity.
            catalog_text = " ".join(TOKEN_RE.findall(self.doc(asin)))
            counts[asin] = sum(
                1 for _tokens, whole in prepared
                if whole and whole in catalog_text
            )
        return counts

    def cumulative_exact_scores(
        self,
        asins: list[str],
        constraints: list[tuple[str, str] | str] | None,
    ) -> dict[str, float]:
        """Return the fraction of active ledger values found exactly in each listing.

        Unlike the old unique-long-phrase override, this deliberately does not require a phrase to
        be rare or four words long. Public sessions frequently disclose several short, shared
        catalog values (for example ``polyester``, ``Imported`` and ``Button closure``). Each
        normalized value contributes at most one point and the denominator is the number of
        distinct active values, yielding a bounded ``[0, 1]`` score. Missing metadata is simply
        unknown and contributes no match or penalty.
        """
        values: list[str] = []
        seen: set[str] = set()
        for item in constraints or []:
            raw = item[1] if isinstance(item, tuple) else item
            canonical = " ".join(TOKEN_RE.findall(str(raw).lower()))
            if not canonical:
                continue
            if canonical not in seen:
                seen.add(canonical)
                values.append(canonical)
        if not values:
            return {asin: 0.0 for asin in asins}
        docs = {
            # Padding gives phrase matching true token boundaries: ``red`` must not match
            # ``redwood`` while ``button closure`` still matches across normalized punctuation.
            asin: " " + " ".join(TOKEN_RE.findall(self.doc(asin).lower())) + " "
            for asin in asins
        }
        total = float(len(values))
        return {
            asin: sum(1 for value in values if f" {value} " in docs[asin]) / total
            for asin in asins
        }

    @staticmethod
    def _raw_ngrams(raw_message: str | None) -> set[str]:
        """Extract stopword-free contiguous bi-grams and tri-grams from one user turn.

        Windows containing a stopword are discarded rather than stitching words across it. This
        preserves true contiguity while removing conversational scaffolding such as ``looking`` /
        ``for`` and ``want``. Catalog punctuation is normalized through ``TOKEN_RE``.
        """
        message = raw_message or ""
        # The evaluator puts the potentially leaked catalog wording after an explicit constraint
        # marker. Prefer that payload so category/scaffolding n-grams (``jewelry necklaces``,
        # ``looking for``) cannot create unrelated promotions; ordinary chat without a marker still
        # uses the complete current turn.
        marker = re.search(
            r"(?:key\s+requirement\s+is|what\s+matters\s+is|what\s+i\s+need\s+is)\s*:\s*(.*)$",
            message,
            re.I,
        )
        if marker:
            message = marker.group(1)
        tokens = [token.lower() for token in TOKEN_RE.findall(message)]
        result: set[str] = set()
        for size in (2, 3):
            for index in range(len(tokens) - size + 1):
                window = tokens[index:index + size]
                if all(len(token) > 1 and token not in STOPWORDS for token in window):
                    result.add(" ".join(window))
        return result

    def raw_ngram_bonus_scores(
        self,
        asins: list[str],
        raw_message: str | None,
        bonus_per_match: float = 0.5,
    ) -> dict[str, float]:
        """Score exact raw-turn bi/tri-gram overlap as a bounded ranking bonus.

        This is intentionally a tie-break signal layered on top of structured cumulative coverage.
        It uses only the current message, never the accumulated transcript, and missing metadata
        remains neutral. A candidate receives ``bonus_per_match`` for each distinct matching
        n-gram.
        """
        ngrams = self._raw_ngrams(raw_message)
        if not ngrams or bonus_per_match <= 0:
            return {asin: 0.0 for asin in asins}
        docs = {
            asin: " " + " ".join(TOKEN_RE.findall(self.doc(asin).lower())) + " "
            for asin in asins
        }
        return {
            asin: bonus_per_match * sum(
                1 for gram in ngrams if f" {gram} " in docs[asin]
            )
            for asin in asins
        }

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
    (lexical=1 dominates); the added semantic term is what survives paraphrase. No popularity re-sort:
    a well-satisfied but unpopular target is no longer demoted.

    Reuses CoverageReranker for the cached catalog text (`doc`), IDF table (`_idf`) and phrase
    tokenization (`_prepare`), so there is one canonical text/IDF source. Semantic similarity comes
    from the shared VectorRetriever (cached product embeddings); if it is unavailable the scorer
    degrades to pure lexical (i.e. coverage).
    """

    def __init__(self, coverage: "CoverageReranker", vector=None,
                 sem_alpha: float = 1.0, pop_weight: float = 0.0,
                 specificity_ref: int = 3,
                 pop_channel: float = 0.7,
                 quality_channel: float = 0.3,
                 sem_gate_low: float = 0.25,
                 sem_gate_high: float = 0.65,
                 unknown_floor: float = 0.5) -> None:
        self._cov = coverage
        self._vector = vector
        self.sem_alpha = sem_alpha
        self.pop_weight = pop_weight
        self.specificity_ref = max(1, specificity_ref)
        total = (pop_channel + quality_channel) or 1.0
        self.pop_channel = pop_channel / total
        self.quality_channel = quality_channel / total
        self.sem_gate_low = max(0.0, min(sem_gate_low, sem_gate_high - 1e-6))
        self.sem_gate_high = max(self.sem_gate_low + 1e-6, sem_gate_high)
        self.unknown_floor = max(0.0, unknown_floor)

    def _lexical(self, toks: list[str], catalog_text: str) -> float:
        """IDF-weighted fraction of the phrase's tokens present verbatim, in [0, 1]."""
        if not toks:
            return 0.0
        total = sum(self._cov._idf(t) for t in toks) or 1.0
        present = sum(self._cov._idf(t) for t in toks if t in catalog_text)
        return present / total

    def score_map(self, asins: list[str], phrases: list[str]) -> tuple[dict[str, float], dict[str, float]]:
        """Compute normalized satisfaction scores without sorting candidates.

        Positive evidence is graded lexical/semantic agreement.  A missing field is UNKNOWN, not
        a conflict: it supplies no negative contribution and leaves the retrieval term to carry a
        sparse listing.  Explicit negative constraints are handled by the existing NeedModel path.
        """
        prepared = self._cov._prepare(phrases)
        zero = {asin: 0.0 for asin in asins}
        if not prepared or len(asins) <= 1:
            return zero, zero
        sims: dict[str, list[float]] = {}
        if self._vector is not None and self.sem_alpha > 0:
            sims = self._vector.phrase_similarity_matrix(
                [whole for _, whole in prepared], asins)
        sat: dict[str, float] = {}
        sem_conf: dict[str, float] = {}
        for asin in asins:
            catalog_text = self._cov.doc(asin)
            row = sims.get(asin)
            num = den = 0.0
            top_sem = 0.0
            for j, (toks, _whole) in enumerate(prepared):
                lex = self._lexical(toks, catalog_text)
                sem = max(0.0, row[j]) if row else 0.0
                if sem > top_sem:
                    top_sem = sem
                match = max(lex, self.sem_alpha * sem)
                weight = 1.0 + COVERAGE_LEN_WEIGHT * len(toks)
                num += weight * match
                den += weight
            raw = num / den if den > 0 else 0.0
            sem_conf[asin] = top_sem
            # Neutral floor: catalog-silent candidates get unknown_floor instead of 0
            sat[asin] = (self.unknown_floor
                         if raw <= 0.0 and top_sem <= 0.0 and self.unknown_floor > 0
                         else raw)
        return sat, sem_conf

    def rank(self, asins: list[str],
             phrases: list[str],
             pop_weight: float | None = None) -> tuple[list[str], dict[str, float]]:
        """Return (ordered_asins, satisfaction_score_per_asin).

        pop_weight overrides the instance weight for this call (0.0 on NL turns).
        """
        eff_pop_weight = self.pop_weight if pop_weight is None else pop_weight
        prepared = self._cov._prepare(phrases)
        if not prepared or len(asins) <= 1:
            return asins, {a: 0.0 for a in asins}
        sat, sem_conf = self.score_map(asins, phrases)
        base_rank = {a: i for i, a in enumerate(asins)}
        ranked = sat
        if eff_pop_weight > 0:
            priors = self._adaptive_prior(asins, sem_conf, len(prepared), pop_weight=eff_pop_weight)
            if priors:
                ranked = {a: sat[a] + priors[a] for a in asins}
        order = sorted(asins, key=lambda a: (-ranked[a], base_rank[a]))
        return order, sat

    def _sem_gate(self, sem_conf: float) -> float:
        lo, hi = self.sem_gate_low, self.sem_gate_high
        if sem_conf >= hi:
            return 0.0
        if sem_conf <= lo:
            return 1.0
        return 1.0 - (sem_conf - lo) / (hi - lo)

    def _quality(self, asin: str) -> float:
        try:
            r = float(self._cov.catalog.get(asin, {}).get("average_rating") or 0.0)
        except (TypeError, ValueError):
            r = 0.0
        return max(0.0, min(1.0, (r - 3.0) / 2.0))

    def _adaptive_prior(self, asins: list[str], sem_conf: dict[str, float],
                        n_phrases: int, pop_weight: float | None = None) -> dict[str, float]:
        if not asins:
            return {}
        eff_pop_weight = self.pop_weight if pop_weight is None else pop_weight
        specificity = min(1.0, n_phrases / self.specificity_ref)
        base_w = eff_pop_weight * (1.0 - specificity)
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


class DualTrackRanker:
    """Single additive fusion of retrieval, satisfaction, cumulative coverage, and a prior.

    The incoming ``asins`` order is the fused BM25+dense retrieval order.  Exact phrase coverage
    is only allowed to dominate when multiple complete phrases identify one candidate and the
    evidence is not shared by the rest of the pool. A separate cumulative term counts active ledger
    values, allowing several shared catalog values to provide useful evidence without requiring one
    artificial four-word unique phrase.
    """

    def __init__(self, coverage: CoverageReranker,
                 satisfaction: NeedSatisfactionScorer) -> None:
        self.coverage = coverage
        self.satisfaction = satisfaction

    @staticmethod
    def _rank_score(index: int, size: int) -> float:
        if size <= 1:
            return 1.0
        return 1.0 - (index / (size - 1))

    @staticmethod
    def _normalize(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        lo, hi = min(values.values()), max(values.values())
        if hi <= lo:
            return {key: 0.0 for key in values}
        return {key: (value - lo) / (hi - lo) for key, value in values.items()}

    @staticmethod
    def _guard_retrieval_head(
        retrieval_order: list[str], ranked_order: list[str], guard_k: int, visible_k: int,
    ) -> tuple[list[str], list[str]]:
        """Keep a bounded retrieval head inside the visible result window.

        Semantic satisfaction is an absolute, noisy signal while retrieval rank is already a fused
        BM25+dense consensus. A small cosine difference must not eject a candidate that retrieval
        placed at rank 1–8 from a ten-item response. Reserve those candidates, replacing the lowest
        unprotected fused candidates, while retaining final-score order within the resulting head.
        The caller only enables this when no complete exact phrase exists, so it cannot dilute the
        catalog-leak path.
        """
        if guard_k <= 0 or visible_k <= 0 or not ranked_order:
            return ranked_order, []
        protected = retrieval_order[:min(guard_k, visible_k, len(retrieval_order))]
        protected_set = set(protected)
        selected = list(ranked_order[:visible_k])
        selected_set = set(selected)
        inserted: list[str] = []
        for asin in protected:
            if asin in selected_set:
                continue
            replace_at = next(
                (index for index in range(len(selected) - 1, -1, -1)
                 if selected[index] not in protected_set),
                None,
            )
            if replace_at is None:
                break
            selected_set.remove(selected[replace_at])
            selected[replace_at] = asin
            selected_set.add(asin)
            inserted.append(asin)
        rank_index = {asin: index for index, asin in enumerate(ranked_order)}
        selected.sort(key=rank_index.__getitem__)
        tail = [asin for asin in ranked_order if asin not in selected_set]
        return selected + tail, inserted

    def _coverage_gate(
        self,
        asins: list[str],
        phrases: list[str],
        exact: dict[str, float],
        min_matches: int,
        discrimination_min: float,
        shared_max: float,
    ) -> tuple[bool, dict[str, float]]:
        counts = self.coverage.exact_match_counts(asins, phrases)
        if not exact or max(exact.values(), default=0.0) <= 0:
            return False, {"top_exact_matches": 0.0, "discrimination": 0.0, "shared_fraction": 1.0}
        top = max(exact, key=exact.get)
        top_score = exact[top]
        ordered = sorted(exact.values())
        # Use the strongest rival rather than the zero-heavy tail. This makes the gate conservative
        # when an exact phrase is common across a category.
        rival_index = max(0, min(len(ordered) - 2, int(len(ordered) * 0.90)))
        rival = ordered[rival_index]
        discrimination = (top_score - rival) / top_score if top_score else 0.0
        shared_fraction = sum(
            1 for value in exact.values() if value >= top_score * 0.80
        ) / max(1, len(exact))
        # Require multiple complete phrases. A single exact phrase can identify the wrong lookalike
        # (for example, a generic ``material alloy`` listing), so it is not enough to turn coverage
        # into the dominant term on its own.
        high = (
            counts.get(top, 0) >= min_matches
            and discrimination >= discrimination_min
            and shared_fraction <= shared_max
        )
        return high, {
            "top_exact_matches": float(counts.get(top, 0)),
            "discrimination": discrimination,
            "shared_fraction": shared_fraction,
        }

    def rank(
        self,
        asins: list[str],
        phrases: list[str],
        *,
        constraints: list[tuple[str, str] | str] | None = None,
        w_ret: float = 1.0,
        w_sat: float = 1.0,
        w_cov_high: float = 2.5,
        w_cov_low: float = 0.0,
        w_leaky: float = 0.0,
        w_legacy_order: float = 0.0,
        w_cumulative: float = 0.0,
        raw_message: str | None = None,
        raw_ngram_bonus: float = 0.0,
        popularity_weight: float = 0.1,
        min_exact_matches: int = 2,
        discrimination_min: float = 0.5,
        shared_max: float = 0.35,
        retrieval_guard_k: int = 0,
        visible_k: int = 10,
        guard_max_exact_matches: int = 0,
    ) -> tuple[list[str], dict[str, dict[str, float] | float | bool]]:
        """Fuse retrieval, satisfaction, structured cumulative coverage, and raw-turn overlap.

        ``raw_message`` is deliberately one turn only. Its exact stopword-free n-gram bonus helps
        resolve equal cumulative-coverage candidates; popularity remains in the fusion as the
        deterministic microscopic tie-break after lexical evidence.
        """
        if len(asins) <= 1:
            return asins, {"coverage_gate": False}

        exact = self.coverage.exact_scores(asins, phrases)
        sat, _sem_conf = self.satisfaction.score_map(asins, phrases)
        # NeedSatisfactionScorer already applies unknown_floor; DualTrackRanker adds its own
        # UNKNOWN guard below for the dual-track context.
        sat = {
            asin: (0.5 if sat.get(asin, 0.0) <= 0.0 and exact.get(asin, 0.0) <= 0.0
                   else sat.get(asin, 0.0))
            for asin in asins
        }
        retrieval = {
            asin: self._rank_score(index, len(asins))
            for index, asin in enumerate(asins)
        }
        pop_raw = {
            asin: self.coverage._pop(asin)
            for asin in asins
        }
        popularity = self._normalize(pop_raw)
        coverage = self._normalize(exact)
        gate, gate_stats = self._coverage_gate(
            asins, phrases, exact, min_exact_matches, discrimination_min, shared_max)
        # ``origin/main``'s public strength came from always sorting on raw verbatim coverage,
        # even when several candidates shared the same boilerplate. Restore that path only when a
        # current-turn disclosed phrase has exact catalog evidence; paraphrased turns have no such
        # evidence and retain the semantic/retrieval track unchanged.
        # A tiny one-token or one-phrase overlap is common in paraphrased honest turns. Require
        # multiple complete disclosed phrases on one candidate before restoring the legacy public
        # leak path; ordinary semantic turns therefore keep the P1/P2 behavior.
        exact_counts = self.coverage.exact_match_counts(asins, phrases)
        leaky_evidence = bool(phrases) and max(exact_counts.values(), default=0) >= 2
        w_cov_gate = w_cov_high if gate else w_cov_low
        w_cov = max(w_cov_gate, w_leaky if leaky_evidence else 0.0)
        legacy_order: list[str] = []
        legacy_rank: dict[str, float] = {asin: 0.0 for asin in asins}
        if leaky_evidence and w_legacy_order > 0:
            # Reuse the main-branch coverage sorter: raw token/phrase coverage first, popularity
            # as tie-break, and RRF with the incoming retrieval order as a bounded floor.
            legacy_order, _legacy_scores = self.coverage.rerank_scored(
                asins,
                phrases,
                pop_blend=COVERAGE_POP_BLEND,
                retrieval_weight=1.0,
            )
            legacy_rank = {
                asin: self._rank_score(index, len(legacy_order))
                for index, asin in enumerate(legacy_order)
            }
        cumulative = self.coverage.cumulative_exact_scores(asins, constraints)
        raw_overlap = self.coverage.raw_ngram_bonus_scores(
            asins, raw_message, bonus_per_match=raw_ngram_bonus)
        final = {
            asin: (
                max(0.0, w_ret) * retrieval[asin]
                + max(0.0, w_sat) * sat.get(asin, 0.0)
                + max(0.0, w_cov) * coverage.get(asin, 0.0)
                + max(0.0, w_legacy_order) * legacy_rank.get(asin, 0.0)
                + max(0.0, w_cumulative) * cumulative.get(asin, 0.0)
                + max(0.0, raw_overlap.get(asin, 0.0))
                + max(0.0, popularity_weight) * popularity.get(asin, 0.0)
            )
            for asin in asins
        }
        base_rank = {asin: index for index, asin in enumerate(asins)}
        order = sorted(asins, key=lambda asin: (-final[asin], base_rank[asin]))
        # On purely paraphrased turns, boundedly preserve the shallow hybrid-retrieval head. This
        # targets observed rank faults where the correct item entered at rank <= 8 but noisy
        # satisfaction pushed it below the evaluator's top-10 cutoff. Any complete exact phrase
        # disables the guard so exact public-set evidence retains full control.
        guard_active = (
            not gate
            and gate_stats["top_exact_matches"] <= max(0, guard_max_exact_matches)
            and retrieval_guard_k > 0
        )
        guarded: list[str] = []
        if guard_active:
            order, guarded = self._guard_retrieval_head(
                asins, order, retrieval_guard_k, visible_k)
        breakdown: dict[str, dict[str, float] | float | bool] = {
            "retrieval": retrieval,
            "satisfaction": sat,
            "coverage": coverage,
            "cumulative_coverage": cumulative,
            "raw_ngram_bonus": raw_overlap,
            "popularity": popularity,
            "final": final,
            "satisfaction_weight": float(max(0.0, w_sat)),
            "coverage_weight": float(w_cov),
            "coverage_gate_weight": float(w_cov_gate),
            "leaky_coverage_active": leaky_evidence,
            "legacy_coverage_order": legacy_rank,
            "legacy_coverage_order_weight": float(max(0.0, w_legacy_order)),
            "cumulative_coverage_weight": float(max(0.0, w_cumulative)),
            "raw_ngram_bonus_per_match": float(max(0.0, raw_ngram_bonus)),
            "coverage_gate": gate,
            "retrieval_guard": bool(guarded),
            "retrieval_guard_count": float(len(guarded)),
            **gate_stats,
        }
        return order, breakdown


def guard_retrieval_head(
    retrieval_order: list[str],
    ranked_order: list[str],
    guard_k: int,
    visible_k: int,
) -> tuple[list[str], list[str]]:
    """Module-level wrapper around DualTrackRanker._guard_retrieval_head for direct import."""
    return DualTrackRanker._guard_retrieval_head(retrieval_order, ranked_order, guard_k, visible_k)


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
