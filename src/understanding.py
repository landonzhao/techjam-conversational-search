"""Deterministic NLU layer.

Regex/vocab-based slot extraction with negation and intensity tracking.
Newer statements supersede conflicting older ones (non-monotonic revision).
No model load, no network calls.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.catalog import TOKEN_RE, text
from src.config import (
    BELIEF_ENTROPY_WEIGHT, BELIEF_MARGIN_WEIGHT, BELIEF_STABILITY_WEIGHT,
    COMPARISON_MARGIN, CONVERGE_HIGH, CONVERGE_MID, SINGLE_VALUED_SLOTS,
    USE_CATEGORY_SWITCH_CLEAR,
)

MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|denim|linen|"
    r"suede|fleece|cashmere|satin|velvet|mesh|down|corduroy|flannel)\b", re.I)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|navy|"
    r"beige|tan|gold|silver|maroon|teal|olive|burgundy)\b", re.I)
STYLE_RE = re.compile(
    r"\b(slim|relaxed|crew|v-neck|vneck|scoop|high-waisted|bootcut|skinny|straight|"
    r"oversized|cropped|hooded|sleeveless|button|turtleneck|a-line|wrap)\b", re.I)
SIZE_RE = re.compile(
    r"\b(xs|s|m|l|xl|xxl|xxxl|small|medium|large|x-?large|xx-?large|petite|plus|"
    r"wide|narrow|size\s*\d{1,2})\b", re.I)
# budget: "under $50", "$40-60", "around $30", "below 25"
BUDGET_RE = re.compile(
    r"(?:under|below|less than|<=?)\s*\$?\s*(\d+(?:\.\d+)?)"
    r"|\$?\s*(\d+)\s*(?:-|to|–)\s*\$?\s*(\d+)"
    r"|(?:around|about|~|budget of)\s*\$?\s*(\d+)", re.I)
# a negated adjective/noun: "not bulky", "no logo", "without pockets", "nothing too heavy"
NEG_FEATURE_RE = re.compile(
    r"\b(?:not|no|without|avoid|nothing|isn'?t|aren'?t|don'?t|too)\s+"
    r"(?:too\s+|so\s+|very\s+)?([a-z]{3,})", re.I)

USE_CASE_KEYS = (
    "hiking", "running", "gym", "workout", "winter", "summer", "beach", "formal",
    "office", "wedding", "work", "casual", "travel", "party", "outdoor", "rain", "sport",
)

# Coarse apparel categories. Grounds the shopper's own head noun ("coat", "sneakers") so
# belief uses the right required-slots. Each surface form maps to a canonical bucket; many
# real head nouns ("handbag", "hobo", "jersey") don't literally contain the base keyword, so
# without synonyms they were missed and an *incidental* noun ("wallet") would win instead.
CATEGORY_CANON: dict[str, str] = {
    # footwear
    "sneaker": "sneaker", "trainer": "sneaker", "kicks": "sneaker",
    "boot": "boot", "bootie": "boot",
    "sandal": "sandal", "flip-flop": "sandal", "flipflop": "sandal",
    "shoe": "shoe", "loafer": "shoe", "moccasin": "shoe", "heel": "shoe",
    # tops / outerwear
    "jacket": "jacket", "parka": "jacket", "windbreaker": "jacket", "blazer": "jacket",
    "coat": "coat", "overcoat": "coat", "raincoat": "coat",
    "hoodie": "hoodie", "sweatshirt": "hoodie",
    "cardigan": "cardigan",
    "sweater": "sweater", "pullover": "sweater", "jumper": "sweater",
    "blouse": "blouse",
    "shirt": "shirt", "tee": "shirt", "t-shirt": "shirt", "tshirt": "shirt",
    "polo": "shirt", "jersey": "shirt", "top": "shirt", "tank": "shirt", "camisole": "shirt",
    # bottoms
    "dress": "dress", "gown": "dress",
    "skirt": "skirt",
    "jeans": "jeans", "denim": "jeans",
    "pants": "pants", "trousers": "pants", "chino": "pants", "scrubs": "pants",
    "shorts": "shorts",
    "leggings": "leggings", "jeggings": "leggings", "tights": "leggings",
    # underwear / sleepwear
    "bra": "bra", "bralette": "bra", "bralet": "bra",
    "nightgown": "sleepwear", "nightshirt": "sleepwear", "nightwear": "sleepwear",
    "pajama": "sleepwear", "pajamas": "sleepwear", "pyjama": "sleepwear", "sleepwear": "sleepwear",
    # accessories
    "socks": "socks", "sock": "socks",
    "scarf": "scarf", "gloves": "gloves", "mittens": "gloves",
    "belt": "belt",
    "backpack": "bag", "bag": "bag", "handbag": "bag", "purse": "bag", "tote": "bag",
    "hobo": "bag", "clutch": "bag", "satchel": "bag", "crossbody": "bag", "wallet": "wallet",
    "watch": "watch",
    "ring": "ring", "necklace": "necklace", "bracelet": "bracelet", "earrings": "earrings",
    "hat": "hat", "cap": "hat", "beanie": "hat", "visor": "hat",
}
# Legacy tuple kept for callers that iterate canonical buckets.
CATEGORY_KEYWORDS = tuple(dict.fromkeys(CATEGORY_CANON.values()))


def resolve_category(low: str) -> str | None:
    """Return the leftmost recognized category surface form, mapped to its canonical bucket.

    Leftmost-wins so the head noun ("handbag") beats an incidental later mention ("wallet pocket").
    """
    best: tuple[int, str] | None = None
    for surface, canon in CATEGORY_CANON.items():
        m = re.search(rf"\b{re.escape(surface)}s?\b", low)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), canon)
    return best[1] if best else None

NEG_CUES = {"not", "no", "without", "avoid", "don't", "dont", "isn't", "isnt",
            "aren't", "arent", "nothing", "never", "less", "too"}
STRONG_CUES = {"must", "need", "needs", "required", "essential", "very", "really", "definitely"}
SOFT_CUES = {"prefer", "ideally", "maybe", "somewhat", "slightly", "kinda", "kind"}
STOP_FEATURE = {"the", "and", "for", "with", "that", "this", "one", "any", "much", "many",
                "them", "does", "was", "are", "but", "not", "too", "very", "really"}


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _ngrams(low: str, n_max: int = 3) -> set[str]:
    """Whole-token 1..n_max-grams of a string, for word-boundary-safe vocab lookup."""
    toks = TOKEN_RE.findall(low)
    grams: set[str] = set()
    for n in range(1, n_max + 1):
        for i in range(len(toks) - n + 1):
            grams.add(" ".join(toks[i:i + n]))
    return grams


@dataclass
class Constraint:
    """One structured, polarity-aware need atom parsed from the conversation."""
    slot: str            # material | color | size | style | budget | use_case | category | feature
    value: str           # normalized value, e.g. "leather", "bulky", "under 50"
    polarity: int = 1    # +1 want, -1 avoid
    weight: float = 1.0  # 1.0 hard ("must"), 0.5 soft ("prefer")
    turn: int = 0

    def key(self) -> tuple[str, str]:
        return (self.slot, _norm(self.value))


@dataclass
class NeedModel:
    """The session's revisable structured understanding of what the shopper wants."""
    constraints: list[Constraint] = field(default_factory=list)
    category: str | None = None

    def revise(self, new: list[Constraint], is_override: bool = False) -> None:
        """Non-monotonic merge (DST selective-overwrite) with surgical correction rules.

        A new constraint on the same (slot, value) supersedes the old one (newer turn wins), so
        'actually, not down' flips a prior 'down'. For SINGLE_VALUED_SLOTS (category/size/budget) a
        new POSITIVE value also supersedes older positive values of that slot — so 'ankle boots' then
        'actually, block-heel sandals' leaves category=sandal, not both. Multi-valued slots
        (color/material/feature/style/use_case) still coexist, so 'black or navy' is preserved.

        Rule (a) — same-turn negation wins: when a positive and negative for the same (slot, value)
        arrive in the same batch (e.g. 'polyester instead of linen'), the negative takes precedence
        and the positive is dropped before any constraint is committed.

        Rule (b) — category-switch retires stale modifiers (only when is_override=True): when the
        router has confirmed an explicit intent override and the category changes, positive non-category
        constraints from prior turns are cleared. Gated on is_override because spurious parser
        category-parses on normal turns caused boundary MTTC regression (+2.5 turns).
        """
        # Rule (a): resolve all same-turn negations before applying positives.
        # Build the set of (slot, value) pairs explicitly negated in this batch.
        same_turn_negative: set[tuple[str, str]] = set()
        for c in new:
            if c.polarity < 0 and c.value:
                same_turn_negative.add((c.slot, c.value))
        # Drop same-turn positives that are explicitly countered by a same-turn negative.
        if same_turn_negative:
            new = [
                c for c in new
                if not (c.polarity > 0 and c.value and (c.slot, c.value) in same_turn_negative)
            ]

        for c in new:
            self.constraints = [x for x in self.constraints if x.key() != c.key()]
            if c.polarity > 0 and c.slot in SINGLE_VALUED_SLOTS:
                self.constraints = [
                    x for x in self.constraints
                    if not (x.slot == c.slot and x.polarity > 0)]

            # Rule (b): category-switch retires prior-turn modifiers — only on confirmed override turns.
            # Gated on is_override so normal turns (where the parser may spuriously extract a category
            # token) never clear valid constraints. USE_CATEGORY_SWITCH_CLEAR can further disable it.
            if (USE_CATEGORY_SWITCH_CLEAR and is_override
                    and c.slot == "category" and c.polarity > 0
                    and self.category and self.category != c.value):
                new_turn = c.turn
                self.constraints = [
                    x for x in self.constraints
                    if x.slot == "category"
                    or x.polarity <= 0
                    or (new_turn is not None and x.turn is not None and x.turn >= new_turn)
                ]

            self.constraints.append(c)
            if c.slot == "category" and c.polarity > 0:
                self.category = c.value

    def positives(self, slot: str | None = None) -> list[Constraint]:
        return [c for c in self.constraints if c.polarity > 0 and (slot is None or c.slot == slot)]

    def negatives(self, slot: str | None = None) -> list[Constraint]:
        return [c for c in self.constraints if c.polarity < 0 and (slot is None or c.slot == slot)]

    def has_positive(self, slot: str) -> bool:
        return any(c.polarity > 0 and c.slot == slot for c in self.constraints)

    def describe(self) -> str:
        """Human-readable dump for chat.py :state."""
        if not self.constraints:
            return "(no structured constraints yet)"
        parts = []
        for c in self.constraints:
            sign = "+" if c.polarity > 0 else "−"
            w = "" if c.weight >= 1.0 else "~"
            parts.append(f"{sign}{w}{c.slot}:{c.value}")
        return "  ".join(parts)


def _clause_before(low: str, start: int, window: int) -> str:
    """Text just before `start`, clipped to the current clause (don't cross , ; . / and/but)
    so a negation in a previous clause doesn't leak into this constraint."""
    pre = low[max(0, start - window):start]
    pre = re.split(r"[,;.]|\band\b|\bbut\b", pre)[-1]
    return pre


def _polarity_near(low: str, start: int, window: int = 28) -> int:
    tail = TOKEN_RE.findall(_clause_before(low, start, window))[-3:]
    return -1 if any(w in NEG_CUES for w in tail) else 1


def _weight_near(low: str, start: int, window: int = 30) -> float:
    tokens = set(TOKEN_RE.findall(_clause_before(low, start, window)))
    return 0.5 if tokens & SOFT_CUES else 1.0


class SlotFiller:
    """Parse a message into polarity-aware `Constraint`s. Deterministic, offline, ~O(len)."""

    def __init__(self, vocab: "CatalogVocab | None" = None) -> None:
        self.vocab = vocab

    def parse(self, text: str, turn: int) -> list[Constraint]:
        low = text.lower()
        out: list[Constraint] = []

        def emit(slot: str, value: str, span: tuple[int, int]) -> None:
            out.append(Constraint(slot, _norm(value),
                                  _polarity_near(low, span[0]),
                                  _weight_near(low, span[0]), turn))

        for rx, slot in ((MATERIAL_RE, "material"), (COLOR_RE, "color"),
                         (STYLE_RE, "style"), (SIZE_RE, "size")):
            for m in rx.finditer(low):
                emit(slot, m.group(1), m.span())

        for key in USE_CASE_KEYS:
            idx = low.find(key)
            if idx >= 0:
                emit("use_case", key, (idx, idx + len(key)))

        bm = BUDGET_RE.search(low)
        if bm:
            if bm.group(1):
                emit("budget", f"under {bm.group(1)}", bm.span())
            elif bm.group(2) and bm.group(3):
                emit("budget", f"{bm.group(2)}-{bm.group(3)}", bm.span())
            elif bm.group(4):
                emit("budget", f"around {bm.group(4)}", bm.span())

        # negated adjectives/nouns that no attribute regex caught (e.g. "not bulky")
        for m in NEG_FEATURE_RE.finditer(low):
            w = m.group(1)
            if w in STOP_FEATURE or MATERIAL_RE.match(w) or COLOR_RE.match(w):
                continue
            out.append(Constraint("feature", _norm(w), -1, 1.0, turn))

        # category = the shopper's head noun (leftmost recognized surface form -> canonical
        # bucket), so "roomy hobo handbag ..." grounds to `bag`, not an incidental `wallet`.
        cat = resolve_category(low)
        if cat:
            out.append(Constraint("category", cat, 1, 1.0, turn))

        # brand via catalog vocab (longest-match), if available
        if self.vocab is not None:
            brand = self.vocab.match_brand(low)
            if brand:
                idx = low.find(brand)
                emit("brand", brand, (idx, idx + len(brand)))

        return out


class CatalogVocab:
    """Vocabulary mined from the catalog: brands, subcategories, price deciles.

    Built once from the loaded catalog dict; no file re-read.
    """

    def __init__(self, brands: set[str], categories: set[str], price_quantiles: list[float]) -> None:
        self.brands = brands
        self.categories = categories
        self.price_quantiles = price_quantiles

    @classmethod
    def build(cls, catalog: dict[str, dict]) -> "CatalogVocab":
        brand_counts: Counter[str] = Counter()
        cats: set[str] = set()
        prices: list[float] = []
        for p in catalog.values():
            store = p.get("store")
            if store:
                brand_counts[_norm(str(store))] += 1
            for c in (p.get("categories") or [])[-2:]:
                token = _norm(str(c))
                if token:
                    cats.add(token)
            price = p.get("price")
            try:
                if price not in (None, ""):
                    prices.append(float(price))
            except (TypeError, ValueError):
                pass
        brands = {b for b, c in brand_counts.items() if c >= 3 and len(b) > 2}
        prices.sort()
        qs = [prices[int(k / 10 * len(prices))] for k in range(1, 10)] if prices else []
        return cls(brands, cats, qs)

    def match_brand(self, low: str) -> str | None:
        hits = _ngrams(low) & self.brands
        return max(hits, key=len) if hits else None


# Seed lexicon of concept synonyms and implications.
# Supplemented at runtime by the data-driven table in cache/synonyms.json.
EXPANSIONS: dict[str, set[str]] = {
    "waterproof": {"gore-tex", "gore tex", "goretex", "water resistant", "water-resistant",
                   "waterproof", "water-repellent", "weatherproof"},
    "warm": {"insulated", "fleece", "wool", "thermal", "sherpa", "down", "cozy", "fluffy"},
    "lightweight": {"light", "lightweight", "breathable", "airy", "packable"},
    "breathable": {"mesh", "moisture-wicking", "wicking", "ventilated", "breathable"},
    "durable": {"rugged", "heavy-duty", "reinforced", "sturdy", "durable"},
    "stretchy": {"spandex", "elastane", "elastic", "flexible", "stretch"},
    "warmth": {"insulated", "fleece", "thermal", "down"},
    "formal": {"dress", "tailored", "smart", "elegant"},
    "casual": {"everyday", "relaxed", "laid-back"},
    "comfortable": {"soft", "cushioned", "comfy", "cozy"},
    "sneaker": {"trainer", "athletic shoe", "running shoe"},
    "jacket": {"coat", "parka", "windbreaker"},
    "grey": {"gray"}, "gray": {"grey"},
    "hiking": {"trail", "outdoor", "trekking"},
    "running": {"jogging", "athletic", "run"},
}


class ExpansionTable:
    """Synonym/implication expansion. Seed lexicon unioned with an optional precomputed
    embedding-neighbor table (cache/synonyms.json). Runtime is pure dict lookup (no encode)."""

    def __init__(self, extra: dict[str, set[str]] | None = None) -> None:
        self.table = {k: set(v) for k, v in EXPANSIONS.items()}
        if extra:
            for k, v in extra.items():
                self.table.setdefault(_norm(k), set()).update(_norm(x) for x in v)

    @classmethod
    def load(cls, path: str = "cache/synonyms.json") -> "ExpansionTable":
        p = Path(path)
        extra = None
        if p.exists():
            try:
                extra = {k: set(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
            except Exception:
                extra = None
        return cls(extra)

    def expand(self, need: NeedModel) -> set[str]:
        """Expansion terms triggered by the shopper's *positive* constraint values."""
        out: set[str] = set()
        for c in need.positives():
            if c.value in self.table:
                out |= self.table[c.value]
            for tok in c.value.split():
                if tok in self.table:
                    out |= self.table[tok]
        return out - {""}

    def expand_text(self, text: str) -> set[str]:
        """Expand from raw query text — catches words the SlotFiller doesn't extract
        as named slots (e.g. 'merino', 'vegan', 'oversized') so the data-driven
        table entries for those terms actually fire."""
        out: set[str] = set()
        low = text.lower()
        # single-token lookup
        for tok in TOKEN_RE.findall(low):
            if tok in self.table:
                out |= self.table[tok]
        # multi-word key lookup (e.g. "business casual", "high waist", "anti odor")
        for key in self.table:
            if " " in key and key in low:
                out |= self.table[key]
        return out - {""}


# Occasion → implied attribute terms ("winter jacket" implies warmth/insulation).
USE_CASE_LEXICON: dict[str, dict[str, set[str]]] = {
    "hiking":  {"terms": {"waterproof", "rugged", "grip", "traction", "gore-tex", "durable"}},
    "trail":   {"terms": {"waterproof", "rugged", "grip", "traction"}},
    "running": {"terms": {"lightweight", "breathable", "cushioned", "athletic"}},
    "jogging": {"terms": {"lightweight", "breathable", "cushioned", "athletic"}},
    "gym":     {"terms": {"moisture-wicking", "spandex", "athletic", "flexible", "breathable"}},
    "workout": {"terms": {"moisture-wicking", "spandex", "athletic", "flexible"}},
    "sport":   {"terms": {"athletic", "breathable", "moisture-wicking"}},
    "winter":  {"terms": {"insulated", "fleece", "wool", "thermal", "warm", "down"}},
    "summer":  {"terms": {"lightweight", "linen", "cotton", "breathable", "quick-dry"}},
    "beach":   {"terms": {"quick-dry", "lightweight", "swim", "uv", "sandal"}},
    "rain":    {"terms": {"waterproof", "water-resistant", "weatherproof", "hooded"}},
    "outdoor": {"terms": {"rugged", "durable", "waterproof", "weatherproof"}},
    "formal":  {"terms": {"dress", "tailored", "slim", "elegant"}},
    "office":  {"terms": {"business", "smart", "tailored", "dress"}},
    "work":    {"terms": {"durable", "slip-resistant", "reinforced"}},
    "wedding": {"terms": {"formal", "elegant", "dress"}},
    "party":   {"terms": {"elegant", "sparkle", "dressy"}},
    "travel":  {"terms": {"packable", "lightweight", "wrinkle-resistant"}},
    "casual":  {"terms": {"relaxed", "everyday", "comfortable"}},
}


class UseCaseInferencer:
    """Infer implied attribute terms from the shopper's stated occasion(s)."""

    def infer(self, need: NeedModel) -> dict:
        cases = {c.value for c in need.positives("use_case")}
        terms: set[str] = set()
        for case in cases:
            entry = USE_CASE_LEXICON.get(case)
            if entry:
                terms |= entry.get("terms", set())
        return {"cases": cases, "terms": terms}


REQUIRED_SLOTS: dict[str, list[str]] = {
    "jacket": ["use_case", "material", "size", "budget"],
    "coat":   ["use_case", "material", "size", "budget"],
    "shoe":   ["size", "use_case", "color"],
    "boot":   ["use_case", "material", "size"],
    "sneaker": ["size", "use_case", "color"],
    "shirt":  ["material", "color", "style", "size"],
    "dress":  ["style", "color", "size", "use_case"],
    "sweater": ["material", "color", "size"],
    "pants":  ["material", "size", "style"],
    "jeans":  ["material", "size", "style"],
    "jewelry": ["material", "color"],
    "ring":   ["material", "color"],
    "necklace": ["material", "color"],
    "watch":  ["material", "color", "style"],
    "bag":    ["material", "color", "style"],
    "hat":    ["material", "color"],
}
DEFAULT_REQ = ["material", "color", "style", "use_case"]


def coarse_category(cat: str | None) -> str | None:
    if not cat:
        return None
    low = cat.lower()
    for key in REQUIRED_SLOTS:
        if key in low:
            return key
    return None


def missing_required(need: NeedModel) -> list[str]:
    req = REQUIRED_SLOTS.get(coarse_category(need.category) or "", DEFAULT_REQ)
    return [s for s in req if not need.has_positive(s)]


def attr_value(product: dict, slot: str, doc: str, price_q: list[float]) -> str | None:
    """The value of a candidate's attribute, for belief distributions / info-gain."""
    if slot == "material":
        m = MATERIAL_RE.search(doc); return m.group(1) if m else None
    if slot == "color":
        m = COLOR_RE.search(doc); return m.group(1) if m else None
    if slot == "style":
        m = STYLE_RE.search(doc); return m.group(1) if m else None
    if slot == "use_case":
        for k in USE_CASE_KEYS:
            if k in doc:
                return k
        return None
    if slot == "budget":
        price = product.get("price")
        try:
            p = float(price)
        except (TypeError, ValueError):
            return None
        bucket = sum(1 for q in price_q if p > q)      # decile bucket 0..9
        return f"q{bucket}"
    return None




# ---------------------------------------------------------------------------
# Backward-compatible re-exports from src.belief
# These names were previously defined here; they now live in src/belief.py.
# Import from src.belief directly in new code.
from src.belief import (  # noqa: E402
    Belief, BeliefModel, converge,
    QuestionSelector, RationaleBuilder,
    apply_negatives, apply_category_gate,
)

__all__ = [
    # NLU core
    "Constraint", "NeedModel", "SlotFiller", "CatalogVocab", "ExpansionTable",
    "UseCaseInferencer", "resolve_category", "coarse_category", "missing_required",
    "attr_value", "REQUIRED_SLOTS", "DEFAULT_REQ",
    # Re-exported from belief.py
    "Belief", "BeliefModel", "converge",
    "QuestionSelector", "RationaleBuilder",
    "apply_negatives", "apply_category_gate",
]
