"""Reranking: Personalizer (popularity + profile tags) then CoverageReranker (verbatim constraints).

Coverage must run last — it resolves verbatim constraint phrases that single out the exact target.
Optional CrossEncoder/LLM rerankers live in src/reranker.py and are fused via RRF before coverage.
"""
from __future__ import annotations

import math
import re

from src.catalog import TOKEN_RE, text, terms
from src.config import (
    COVERAGE_FULL_PHRASE_BONUS, COVERAGE_LEN_WEIGHT, COVERAGE_TIE_BREAK,
    POP_WEIGHT, TAG_WEIGHT,
)


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

    def _coverage(self, asin: str, phrases: list[tuple[list[str], str]]) -> float:
        catalog_text = self.doc(asin)
        score = 0.0
        for toks, whole in phrases:
            present = sum(1 for t in toks if t in catalog_text)
            weight = 1.0 + COVERAGE_LEN_WEIGHT * len(toks)
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
    ) -> tuple[list[str], dict[str, float]]:
        """Rerank and return (ordered_list, coverage_score_per_asin).

        prefer_cat (measured −0.019; available for ablation): on a tie, prefer the candidate
        whose title matches the shopper's category before falling back to popularity.
        """
        prepared = self._prepare(phrases)
        if not prepared or not asins:
            return asins, {}
        scores = {a: self._coverage(a, prepared) for a in asins}
        base_rank = {a: i for i, a in enumerate(asins)}
        if prefer_cat:
            cm = {a: self._cat_match(a, prefer_cat) for a in asins}
            key = lambda a: (-scores[a], -cm[a], -self._pop(a), base_rank[a])
        elif COVERAGE_TIE_BREAK == "base":
            key = lambda a: (-scores[a], base_rank[a])
        else:  # "pop" — popularity tie-break
            key = lambda a: (-scores[a], -self._pop(a), base_rank[a])
        return sorted(asins, key=key), scores

    def rerank(self, asins: list[str], phrases: list[str],
               prefer_cat: str | None = None) -> list[str]:
        return self.rerank_scored(asins, phrases, prefer_cat)[0]

    def _cat_match(self, asin: str, cat: str) -> int:
        # title only — the categories field is polluted by Amazon's top-level path
        title = text(self.catalog.get(asin, {}).get("title")).lower()
        return 1 if re.search(rf"\b{re.escape(cat)}s?\b", title) else 0
