"""Belief model, convergence policy, clarification strategy, and rationale building.

This module owns everything that operates on *ranked candidates* and the agent's explicit
belief over the search state. It depends on the NLU layer (src/understanding.py) but not
vice versa, keeping the dependency arrow clean:

    src/belief.py  →  src/understanding.py  →  src/catalog.py
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from src.catalog import text
from src.config import (
    BELIEF_ENTROPY_WEIGHT, BELIEF_MARGIN_WEIGHT, BELIEF_STABILITY_WEIGHT,
    COMPARISON_MARGIN, CONVERGE_HIGH, CONVERGE_MID,
)
from src.understanding import (
    NeedModel, CatalogVocab,
    attr_value, coarse_category, resolve_category,
    REQUIRED_SLOTS, DEFAULT_REQ,
)


# ---------------------------------------------------------------------------
# Belief state

@dataclass
class Belief:
    """The agent's explicit, inspectable model of the need + its own confidence."""
    top_asin: str | None = None
    margin: float = 0.0
    entropy: float = 0.0
    stable_turns: int = 0
    category: str | None = None
    item_confidence: float = 0.0
    need_confidence: float = 0.0
    confidence: float = 0.0
    attr_uncertainty: dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        unc = " ".join(f"{s}:{u:.2f}" for s, u in self.attr_uncertainty.items())
        return (f"conf={self.confidence:.2f} (item={self.item_confidence:.2f} "
                f"need={self.need_confidence:.2f}) top={self.top_asin} "
                f"margin={self.margin:.2f} stable={self.stable_turns} cat={self.category} "
                f"| uncertainty[{unc}]")


# ---------------------------------------------------------------------------
# Belief model

class BeliefModel:
    """Turns the ranked pool + scores into an attribute-level belief over the need."""

    TOPN = 20

    def __init__(self, catalog: dict[str, dict], doc_fn, vocab: CatalogVocab) -> None:
        self.catalog = catalog
        self.doc = doc_fn
        self.price_q = vocab.price_quantiles

    def _modal_category(self, head: list[str]) -> str | None:
        from src.understanding import _norm
        counts: Counter[str] = Counter()
        for a in head:
            cats = self.catalog.get(a, {}).get("categories") or []
            if cats:
                counts[_norm(str(cats[-1]))] += 1
        return counts.most_common(1)[0][0] if counts else None

    def update(self, order: list[str], scores: dict[str, float],
               need: NeedModel, prev: Belief | None) -> Belief:
        head = order[:self.TOPN]
        vals = [max(scores.get(a, 0.0), 0.0) for a in head]
        s0 = vals[0] if vals else 0.0
        margin = (s0 - vals[-1]) / s0 if s0 > 0 else 0.0
        tot = sum(vals)
        ent = 0.0
        if tot > 0:
            ps = [v / tot for v in vals if v > 0]
            if len(ps) > 1:
                ent = -sum(p * math.log(p) for p in ps) / math.log(len(ps))
        top = head[0] if head else None
        stable = (prev.stable_turns + 1) if (prev and top and prev.top_asin == top) else 0

        cat = need.category or self._modal_category(head)
        req = REQUIRED_SLOTS.get(coarse_category(cat) or "", DEFAULT_REQ)
        miss = [s for s in req if not need.has_positive(s)]
        attr_unc: dict[str, float] = {}
        for slot in miss:
            counts: Counter[str] = Counter()
            for a in head:
                v = attr_value(self.catalog.get(a, {}), slot, self.doc(a), self.price_q)
                if v:
                    counts[v] += 1
            if counts:
                t = sum(counts.values())
                ps = [c / t for c in counts.values()]
                u = -sum(p * math.log(p) for p in ps) / math.log(len(ps)) if len(ps) > 1 else 0.0
                attr_unc[slot] = max(u, 0.5)
            else:
                attr_unc[slot] = 1.0

        need_conf = 1.0 - (sum(attr_unc.values()) / len(attr_unc) if attr_unc else 0.0)
        item_conf = (BELIEF_MARGIN_WEIGHT * margin
                     + BELIEF_ENTROPY_WEIGHT * (1.0 - ent)
                     + BELIEF_STABILITY_WEIGHT * min(stable / 2.0, 1.0))
        conf = min(item_conf, need_conf)
        return Belief(top, margin, ent, stable, cat, item_conf, need_conf, conf, attr_unc)


# ---------------------------------------------------------------------------
# Convergence policy

def converge(belief: Belief, missing: list[str], turn: int, last_turn: int = 10) -> str:
    """Return DELIVER, CONFIRM, or PROBE based on current belief and turn."""
    if belief.confidence >= CONVERGE_HIGH or turn >= last_turn:
        return "DELIVER"
    if belief.item_confidence >= CONVERGE_MID and not missing:
        return "CONFIRM"
    return "PROBE"


# ---------------------------------------------------------------------------
# Clarification strategy

# Slot decision weights: how much resolving each slot narrows the candidate pool.
DECISION_WEIGHT = {"budget": 1.3, "size": 1.2, "material": 1.1, "use_case": 1.0,
                   "category": 1.0, "style": 0.9, "color": 0.8}

_EXTRACTED_SLOTS = {"material", "color", "style", "use_case", "size"}
_WORD_RE = re.compile(r"[a-z0-9]+")
_FACET_STOP = {
    "with", "and", "for", "the", "this", "that", "from", "your", "you", "our", "are", "all",
    "womens", "women", "mens", "men", "kids", "girls", "boys", "unisex", "size", "sizes",
    "small", "medium", "large", "pack", "set", "pair", "new", "style", "fashion", "quality",
    "premium", "classic", "made", "design", "designed", "perfect", "great", "features", "product",
    "material", "color", "colors", "available", "please", "will", "can", "has", "have",
}


class QuestionSelector:
    """Ask the question that most reduces belief uncertainty or confirms the top hypothesis."""

    adaptive_clarify = False

    def __init__(self, catalog: dict[str, dict], doc_fn, price_q: list[float]) -> None:
        self.catalog = catalog
        self.doc = doc_fn
        self.price_q = price_q

    def select(self, belief: Belief, need: NeedModel, conv_state: str,
               head: list[str], guidance: dict[str, float] | None = None,
               turn: int = 0) -> tuple[str | None, str]:
        if conv_state == "DELIVER":
            return None, "Here are the closest matches based on what you've told me."
        if conv_state == "CONFIRM" and belief.top_asin:
            attr = self._distinctive_attr(belief.top_asin, head)
            if attr:
                return attr, self._confirm_phrase(attr, belief.top_asin)
        # Stage-aware clarification (SIGIR 2026): item-based questions from turn 4+
        if turn >= 4 and len(head) >= 2:
            cmp = self._comparison_phrase(head)
            if cmp:
                return "other", cmp
        if belief.margin < COMPARISON_MARGIN and len(head) >= 2:
            cmp = self._comparison_phrase(head)
            if cmp:
                return "other", cmp
        unc = dict(belief.attr_uncertainty)
        facet_word: str | None = None
        if self.adaptive_clarify:
            unc = {s: u for s, u in unc.items()
                   if s not in _EXTRACTED_SLOTS or self._top_values(head, s)}
            if not need.has_positive("feature"):
                facet = self._feature_facet(head, need)
                if facet:
                    facet_word, unc["feature"] = facet[0], facet[1]
        if unc:
            g = guidance or {}
            attr = max(unc, key=lambda s: unc[s] * DECISION_WEIGHT.get(s, 1.0) * g.get(s, 1.0))
            if attr == "feature" and facet_word:
                return "feature", f"Any particular feature that matters — like {facet_word}?"
            return attr, self._probe_phrase(attr, head)
        return "other", "Is there a specific detail that matters most to you?"

    def discovery_message(self, head: list[str], category: str | None) -> str | None:
        """Discovery Mode (CoShop/CoPref 2026): present 3 product archetypes from the pool."""
        if len(head) < 3:
            return None
        n = len(head)
        indices = [0, max(1, n // 3), max(2, 2 * n // 3)]
        picks = [head[i] for i in indices if i < n]
        if len(picks) < 2:
            return None
        lines: list[str] = []
        for asin in picks:
            p = self.catalog.get(asin, {})
            title = str(p.get("title") or "")[:55]
            price = p.get("price")
            desc = f"**{title}**"
            if price:
                try:
                    desc += f" (~${float(price):.0f})"
                except (TypeError, ValueError):
                    pass
            lines.append(f"• {desc}")
        cat_phrase = f" for {category}" if category else ""
        intro = f"I found a few different directions{cat_phrase}:"
        outro = "Which of these resonates, or would you like something different?"
        return "\n".join([intro] + lines + [outro])

    def _feature_facet(self, head: list[str], need: NeedModel) -> tuple[str, float] | None:
        pool = head[:12]
        if len(pool) < 4:
            return None
        known = {t for c in need.positives() for t in _WORD_RE.findall(c.value.lower())}
        docs = [set(_WORD_RE.findall(self.doc(a))) for a in pool]
        counts: Counter[str] = Counter()
        for d in docs:
            counts.update(t for t in d if len(t) > 3 and t not in _FACET_STOP and t not in known)
        n = len(pool)
        best: tuple[str, float] | None = None
        for tok, c in counts.items():
            if c < 2 or c > n - 2:
                continue
            frac = c / n
            strength = 1.0 - abs(0.5 - frac) * 2.0
            if strength >= 0.5 and (best is None or strength > best[1]):
                best = (tok, strength)
        return best

    def _top_values(self, head: list[str], slot: str, n: int = 3) -> list[str]:
        counts: Counter[str] = Counter()
        for a in head:
            v = attr_value(self.catalog.get(a, {}), slot, self.doc(a), self.price_q)
            if v:
                counts[v] += 1
        return [v for v, _ in counts.most_common(n)]

    def _price_range(self, head: list[str]) -> tuple[float, float] | None:
        prices = []
        for a in head:
            try:
                prices.append(float(self.catalog.get(a, {}).get("price")))
            except (TypeError, ValueError):
                pass
        return (min(prices), max(prices)) if prices else None

    def _probe_phrase(self, attr: str, head: list[str]) -> str:
        if attr == "budget":
            pr = self._price_range(head)
            if pr:
                return f"These range from ${pr[0]:.0f} to ${pr[1]:.0f} — do you have a budget in mind?"
            return "What's your budget?"
        if attr in ("color", "material", "style"):
            vals = self._top_values(head, attr)
            if vals:
                return f"I'm seeing {', '.join(vals)} — any {attr} preference?"
            return {"color": "Any color preference?", "material": "Any material preference?",
                    "style": "What style are you after?"}[attr]
        if attr == "category":
            cats = self._top_values(head, "category") or []
            if len(cats) >= 2:
                return f"Are you leaning toward {cats[0]} or {cats[1]}?"
            return "What type of item are you after?"
        if attr == "size":
            return "What size do you need?"
        if attr == "use_case":
            return "What will you mainly use it for?"
        return "Is there a specific detail that matters most to you?"

    def _distinctive_attr(self, top: str, head: list[str]) -> str | None:
        runners = [a for a in head if a != top][:5]
        for slot in ("material", "color", "style", "category"):
            tv = attr_value(self.catalog.get(top, {}), slot, self.doc(top), self.price_q)
            if not tv:
                continue
            rvs = [attr_value(self.catalog.get(r, {}), slot, self.doc(r), self.price_q) for r in runners]
            if any(rv and rv != tv for rv in rvs):
                return slot
        return None

    def _confirm_phrase(self, attr: str, top: str) -> str:
        tv = attr_value(self.catalog.get(top, {}), attr, self.doc(top), self.price_q)
        return f"The closest match is {tv} — is that what you want, or something different?"

    def _comparison_phrase(self, head: list[str]) -> str | None:
        a1, a2 = head[0], head[1]
        t1 = str(self.catalog.get(a1, {}).get("title") or "")
        t2 = str(self.catalog.get(a2, {}).get("title") or "")
        if not t1 or not t2 or t1[:40] == t2[:40]:
            return None
        p1 = self.catalog.get(a1, {}).get("price")
        p2 = self.catalog.get(a2, {}).get("price")
        desc1 = f"{t1[:50]}" + (f" (${p1:.0f})" if p1 else "")
        desc2 = f"{t2[:50]}" + (f" (${p2:.0f})" if p2 else "")
        return f"Are you looking for something more like '{desc1}' or '{desc2}'?"


# ---------------------------------------------------------------------------
# Rationale building

class RationaleBuilder:
    """Builds match rationale strings: constraint coverage, snippet evidence, and contrast."""

    def __init__(self, catalog: dict[str, dict], doc_fn, vector=None) -> None:
        self.catalog = catalog
        self.doc = doc_fn
        self._vector = vector

    def _within_budget(self, asin: str, value: str) -> bool:
        m = re.search(r"(\d+)", value)
        if not m:
            return False
        try:
            return float(self.catalog.get(asin, {}).get("price")) <= float(m.group(1))
        except (TypeError, ValueError):
            return False

    def build(self, asin: str, need: NeedModel) -> str:
        doc = self.doc(asin)
        hits: list[str] = []
        for c in need.positives():
            if c.slot == "category":
                continue
            if c.slot == "budget":
                if self._within_budget(asin, c.value):
                    hits.append(c.value.replace("under ", "under $").replace("around ", "around $"))
                continue
            if c.value and all(tok in doc for tok in c.value.split()):
                hits.append(c.value)
        seen: list[str] = []
        for h in hits:
            if h not in seen:
                seen.append(h)
        return "matches " + ", ".join(seen[:4]) if seen else ""

    def build_snippet(self, asin: str, need: NeedModel) -> str | None:
        """Best-matching description sentence by BGE cosine. (Snippet-CRS, 2024)"""
        if self._vector is None:
            return None
        desc = str(self.catalog.get(asin, {}).get("description") or "")
        sentences = [s.strip() for s in re.split(r"[.!?]", desc) if len(s.strip()) > 20]
        if not sentences:
            return None
        phrases = [c.value for c in need.positives() if c.value and c.slot != "budget"]
        if not phrases:
            return None
        query = " ".join(phrases[:4])
        try:
            sims = self._vector.phrase_similarities([query], sentences)
            if not sims:
                return None
            best = max(sims, key=sims.get)
            if sims[best] < 0.25:
                return None
            return best
        except Exception:
            return None

    def build_contrast(self, asin1: str, asin2: str, need: NeedModel) -> str | None:
        """Slot-level differential between top-2 candidates. (C2-CRS, WSDM 2022)"""
        p1 = self.catalog.get(asin1, {})
        p2 = self.catalog.get(asin2, {})
        t1 = str(p1.get("title") or "")[:40]
        t2 = str(p2.get("title") or "")[:40]
        if not t1 or not t2:
            return None
        doc1, doc2 = self.doc(asin1), self.doc(asin2)
        positives = [c for c in need.positives() if c.slot not in ("category", "budget") and c.value]
        a_wins, b_wins = [], []
        for c in positives[:3]:
            in1 = c.value.lower() in doc1.lower()
            in2 = c.value.lower() in doc2.lower()
            if in1 and not in2:
                a_wins.append(c.value)
            elif in2 and not in1:
                b_wins.append(c.value)
        price_note = ""
        try:
            pr1, pr2 = float(p1.get("price") or 0), float(p2.get("price") or 0)
            if pr1 > 0 and pr2 > 0 and abs(pr1 - pr2) >= 5:
                cheaper = t1 if pr1 < pr2 else t2
                price_note = f"{cheaper} is ${abs(pr1 - pr2):.0f} cheaper"
        except (TypeError, ValueError):
            pass
        parts = []
        if a_wins:
            parts.append(f"{t1} better matches your {', '.join(a_wins[:2])}")
        if b_wins:
            parts.append(f"{t2} better matches your {', '.join(b_wins[:2])}")
        if price_note:
            parts.append(price_note)
        if not parts:
            return None
        return "Comparing top picks: " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Candidate filtering helpers

def apply_negatives(candidates: list[str], need: NeedModel, doc_fn) -> list[str]:
    """Demote avoid-constraint violators to the back without dropping any candidates."""
    negs = [c.value for c in need.negatives() if c.value]
    if not negs or not candidates:
        return candidates
    keep, drop = [], []
    for asin in candidates:
        t = doc_fn(asin)
        (drop if any(v in t for v in negs) else keep).append(asin)
    return keep + drop


def apply_category_gate(
    candidates: list[str], need_category: str | None, catalog: dict[str, dict]
) -> list[str]:
    """Demote candidates whose title resolves to a different category than the stated need."""
    if not need_category or not candidates:
        return candidates
    keep, demote = [], []
    for asin in candidates:
        title = text(catalog.get(asin, {}).get("title")).lower()
        cand_cat = resolve_category(title)
        (demote if (cand_cat and cand_cat != need_category) else keep).append(asin)
    return keep + demote
