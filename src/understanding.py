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

from src.catalog import TOKEN_RE
from src.config import CONVERGE_HIGH, CONVERGE_MID

MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|denim|linen|"
    r"suede|fleece|cashmere|satin|velvet|mesh|down|corduroy|flannel)\b", re.I)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|navy|"
    r"beige|tan|gold|silver|maroon|teal|olive|burgundy)\b", re.I)
STYLE_RE = re.compile(
    r"\b(slim|relaxed|crew|v-neck|vneck|scoop|high-waisted|bootcut|skinny|straight|"
    r"oversized|cropped|hooded|sleeveless|button|turtleneck|a-line|wrap)\b", re.I)
FEATURE_RE = re.compile(
    r"\b(waterproof|water[- ]resistant|breathable|comfortable|comfy|warm|lightweight|"
    r"durable|stretchy|machine washable|insulated|cushioned|packable)\b", re.I)
# Apostrophes are non-word characters to Python's ``\b``.  Without the explicit guards below,
# the ``m`` in ``I'm`` and the ``s`` in ``Valentine's`` look like standalone sizes.  Keep the
# single-letter sizes useful (``size m``, ``m``), but never accept a token adjacent to an ASCII or
# curly apostrophe.
SIZE_RE = re.compile(
    r"(?<!['’])\b(size\s*\d{1,2}|xxxl|xxl|xl|xs|x-?large|xx-?large|small|medium|large|"
    r"petite|plus|wide|narrow|s|m|l)\b(?!['’])", re.I)
# budget: "under $50", "$40-60", "around $30", "below 25"
BUDGET_RE = re.compile(
    r"(?:under|below|less than|<=?)\s*\$?\s*(\d+(?:\.\d+)?)"
    r"|\$?\s*(\d+)\s*(?:-|to|–)\s*\$?\s*(\d+)"
    r"|(?:around|about|~|budget of)\s*\$?\s*(\d+)", re.I)
# a negated adjective/noun: "not bulky", "no logo", "without pockets", "nothing too heavy"
NEG_FEATURE_RE = re.compile(
    r"\b(?:not|no|without|avoid|remove|drop|ditch|skip|exclude|nothing|isn'?t|aren'?t|don'?t|too)\s+"
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
    "slipper": "shoe",
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
    "bucket hat": "hat", "bucket": "hat",
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


def _category_matches(low: str) -> list[tuple[int, int, str, str]]:
    """Return non-overlapping category mentions as (start, end, surface, canonical).

    At the same position the longest surface wins (``bucket hat`` over ``bucket``).
    """
    found: list[tuple[int, int, str, str]] = []
    for surface, canon in CATEGORY_CANON.items():
        for match in re.finditer(rf"\b{re.escape(surface)}s?\b", low):
            found.append((match.start(), match.end(), match.group(0), canon))
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    result: list[tuple[int, int, str, str]] = []
    for item in found:
        if result and item[0] < result[-1][1]:
            continue
        result.append(item)
    return result

NEG_CUES = {"not", "no", "without", "avoid", "remove", "drop", "ditch", "skip", "exclude",
            "don't", "dont", "isn't", "isnt", "aren't", "arent", "nothing", "never",
            "less", "too"}
STRONG_CUES = {"must", "need", "needs", "required", "essential", "very", "really", "definitely"}
SOFT_CUES = {"prefer", "ideally", "maybe", "somewhat", "slightly", "kinda", "kind"}
STOP_FEATURE = {"the", "and", "for", "with", "that", "this", "one", "any", "much", "many",
                "them", "does", "was", "are", "but", "not", "too", "very", "really",
                # These are request verbs, not rejected product features.  Without this guard
                # ``don't want linen`` can produce a spurious ``feature:want`` event.
                "want", "need", "like", "prefer"}


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
    """One ordered preference-ledger event parsed from the conversation."""
    slot: str            # material | color | size | style | budget | use_case | category | feature
    value: str           # normalized value, e.g. "leather", "bulky", "under 50"
    polarity: int = 1    # +1 want, -1 avoid, 0 for CLEAR / NO_PREFERENCE
    weight: float = 1.0  # 1.0 hard ("must"), 0.5 soft ("prefer")
    turn: int = 0
    operation: str | None = None  # SET | ADD | REMOVE | CLEAR | NO_PREFERENCE
    span: tuple[int, int] | None = None
    surface: str | None = None
    source: str = "regex"
    confidence: float = 1.0
    active: bool = True

    def key(self) -> tuple[str, str]:
        return (self.slot, _norm(self.value))


@dataclass
class NeedModel:
    """The session's revisable structured understanding of what the shopper wants."""
    constraints: list[Constraint] = field(default_factory=list)
    category: str | None = None
    ledger: list[Constraint] = field(default_factory=list)
    no_preference: set[str] = field(default_factory=set)

    EXCLUSIVE_SLOTS = frozenset({"category", "brand", "color", "size", "budget"})
    CATEGORY_MODIFIER_SLOTS = frozenset({
        "material", "color", "style", "feature", "brand", "size", "budget",
    })
    VALID_OPERATIONS = frozenset({"SET", "ADD", "REMOVE", "CLEAR", "NO_PREFERENCE"})

    def _deactivate(self, predicate) -> None:
        for old in self.constraints:
            if predicate(old):
                old.active = False
        self.constraints = [old for old in self.constraints if old.active]

    def _last_active_slot(self) -> str | None:
        for event in reversed(self.ledger):
            if event.active and event.slot != "__last__":
                return event.slot
        return None

    def _refresh_category(self) -> None:
        cats = self.positives("category")
        self.category = cats[-1].value if cats else None

    def revise(self, new: list[Constraint]) -> None:
        """Apply ordered SET/ADD/REMOVE/CLEAR/NO_PREFERENCE ledger operations.

        ``constraints`` remains the backwards-compatible active view consumed by ranking;
        ``ledger`` retains every event, including superseded events, for audit and query masking.
        """
        # A parser/LLM pair may call revise twice for one turn.  Resolve all explicit removals
        # before applying positives so a later fallback extraction cannot resurrect a rejected
        # value (for example ``instead of linen ... polyester``).  The same filter also handles a
        # single extraction containing both polarity events, regardless of their order.
        same_turn_negative: set[tuple[str, str]] = {
            (_norm(c.slot), _norm(c.value))
            for c in new
            if c.value and (c.polarity < 0 or (c.operation or "").upper() == "REMOVE")
        }
        if new:
            turn = new[0].turn
            # ``turn=0`` is the backwards-compatible default used by callers that apply one
            # operation at a time without session turn metadata.  Only consult prior ledger
            # events when a real turn id is available; events in this `new` batch are still
            # protected even at turn zero.
            if turn != 0:
                same_turn_negative.update(
                    (_norm(old.slot), _norm(old.value))
                    for old in self.ledger
                    if old.turn == turn and old.value
                    and (old.polarity < 0 or (old.operation or "").upper() == "REMOVE")
                )
            new = [
                c for c in new
                if not (c.value and c.polarity > 0
                        and (c.operation or "").upper() != "REMOVE"
                        and (_norm(c.slot), _norm(c.value)) in same_turn_negative)
            ]

        for c in new:
            c.value = _norm(c.value)
            op = (c.operation or (
                "REMOVE" if c.polarity < 0
                else "SET" if c.slot in self.EXCLUSIVE_SLOTS
                else "ADD"
            )).upper()
            if op not in self.VALID_OPERATIONS:
                op = "ADD"
            c.operation = op

            if c.slot == "__last__" and op in {"CLEAR", "NO_PREFERENCE"}:
                c.slot = self._last_active_slot() or "__last__"

            if op in {"CLEAR", "NO_PREFERENCE"}:
                if c.slot == "__modifiers__":
                    self._deactivate(lambda old: old.slot in self.CATEGORY_MODIFIER_SLOTS)
                else:
                    self._deactivate(lambda old, slot=c.slot: old.slot == slot)
                c.polarity = 0
                c.active = False
                if op == "NO_PREFERENCE" and c.slot != "__last__":
                    self.no_preference.add(c.slot)
                elif c.slot != "__last__":
                    self.no_preference.discard(c.slot)
                self.ledger.append(c)
                self._refresh_category()
                continue

            self.no_preference.discard(c.slot)
            previous_category = self.category
            if op == "SET":
                # A new positive exclusive value supersedes other positive values. Explicit
                # negatives for *different* values remain useful ("not red, blue").
                self._deactivate(
                    lambda old, cur=c: old.slot == cur.slot
                    and (old.polarity > 0 or old.key() == cur.key()))
                c.polarity = 1
            elif op == "ADD":
                self._deactivate(lambda old, cur=c: old.key() == cur.key())
                c.polarity = 1
            else:  # REMOVE
                self._deactivate(lambda old, cur=c: old.key() == cur.key())
                c.polarity = -1

            c.active = True
            self.constraints.append(c)
            self.ledger.append(c)
            self._refresh_category()

            if (c.slot == "category" and c.polarity > 0
                    and previous_category and previous_category != c.value):
                # A category transition starts a new product branch. Retire positive modifiers
                # learned before this switch (including prior turns and earlier spans in this
                # turn); use-case intent is deliberately retained because it can carry across
                # footwear transitions (running shoes) and is handled separately below.
                switch_start = c.span[0] if c.span else None
                prior_category_start = None
                if switch_start is not None:
                    same_turn = [
                        old for old in self.ledger
                        if old.slot == "category" and old.turn == c.turn
                        and old.value == previous_category and old.span
                        and old.span[0] < switch_start
                    ]
                    if same_turn:
                        prior_category_start = min(old.span[0] for old in same_turn)
                self._deactivate(
                    lambda old: old.slot in self.CATEGORY_MODIFIER_SLOTS
                    and old.polarity > 0
                    and (old.turn < c.turn or switch_start is None
                         or (prior_category_start is not None and old.span
                             and old.span[0] < switch_start)))
                self._refresh_category()

            if (c.slot == "category" and c.polarity > 0 and c.value == "shoe"
                    and previous_category == "boot"):
                # Category-dependent context from the abandoned boot branch must not survive a
                # footwear switch; a subsequent running preference can then stand on its own.
                self._deactivate(
                    lambda old: old.slot == "use_case"
                    and old.value == "hiking" and old.polarity > 0)
                self._refresh_category()

            # Running is a strong footwear correction signal. In noisy speech a shopper may say
            # "running kind" or "running does" without repeating "shoes"; when the active
            # category is boots, infer the canonical shoe switch and retire hiking (the old boot
            # use-case) as well. Explicit "running shoes" is still handled by the category parser.
            if (c.slot == "use_case" and c.polarity > 0 and c.value == "running"
                    and previous_category == "boot" and self.category == "boot"):
                inferred = Constraint(
                    "category", "shoe", 1, c.weight, c.turn,
                    operation="SET", source="regex_inferred", confidence=0.85,
                )
                self.revise([inferred])
                self._deactivate(
                    lambda old: old.slot == "use_case"
                    and old.value == "hiking" and old.polarity > 0)
                self._refresh_category()

        # When both signals occur in one noisy turn ("running shoes like hiking boots"), the
        # running-shoe request is the final footwear intent; do not leave the boot use-case alive.
        if (any(c.slot == "use_case" and c.value == "running" and c.polarity > 0
                for c in new) and self.category == "shoe"
                and any(c.slot == "use_case" and c.value == "running" and c.polarity > 0
                        for c in self.constraints)):
            self._deactivate(
                lambda old: old.slot == "use_case"
                and old.value == "hiking" and old.polarity > 0)

    def active_signature(self) -> tuple[tuple[str, str, int], ...]:
        return tuple((c.slot, c.value, c.polarity) for c in self.constraints)

    def excluded_terms(self) -> set[str]:
        """Terms explicitly rejected or superseded and not active as a positive value."""
        active = {c.value for c in self.positives()}
        excluded: set[str] = set()
        for event in self.ledger:
            if event.value and (not event.active or event.polarity <= 0):
                excluded.add(_norm(event.surface or event.value))
                excluded.add(event.value)
        return {term for term in excluded if term and term not in active}

    def positives(self, slot: str | None = None) -> list[Constraint]:
        return [c for c in self.constraints if c.polarity > 0 and (slot is None or c.slot == slot)]

    def negatives(self, slot: str | None = None) -> list[Constraint]:
        return [c for c in self.constraints if c.polarity < 0 and (slot is None or c.slot == slot)]

    def has_positive(self, slot: str) -> bool:
        return any(c.polarity > 0 and c.slot == slot for c in self.constraints)

    def describe(self) -> str:
        """Human-readable dump for chat.py :state."""
        if not self.constraints and not self.no_preference:
            return "(no structured constraints yet)"
        parts = []
        for c in self.constraints:
            sign = "+" if c.polarity > 0 else "−"
            w = "" if c.weight >= 1.0 else "~"
            parts.append(f"{sign}{w}{c.slot}:{c.value}")
        parts.extend(f"∅{slot}:no-preference" for slot in sorted(self.no_preference))
        return "  ".join(parts)


def _clause_before(low: str, start: int, window: int) -> str:
    """Text just before `start`, clipped to the current clause (don't cross , ; . / and/but)
    so a negation in a previous clause doesn't leak into this constraint."""
    pre = low[max(0, start - window):start]
    pre = re.split(
        r"[,;.]|\band\b|\bbut\b|\b(?:actually|actly|instead|rather|"
        r"scratch that|never ?mind|changed my mind|make that|make it|i mean|"
        r"wait(?:\s+(?:no|nah))?|nah)\b",
        pre,
    )[-1]
    return pre


def _polarity_near(low: str, start: int, window: int = 28) -> int:
    # Two-token scope prevents a repair marker before an intervening replacement noun from
    # leaking into the next attribute: "no, shoes for running" wants running, not avoids it.
    tail = TOKEN_RE.findall(_clause_before(low, start, window))[-2:]
    return -1 if any(w in NEG_CUES for w in tail) else 1


_EXPLICIT_NEGATION_BEFORE_RE = re.compile(
    r"(?:"
    r"(?:instead\s+of|rather\s+than)(?:\s+(?:a|an|any|the))?"
    r"|(?:no|not|without|avoid|remove|drop|ditch|skip|exclude)"
    r"(?:\s+(?:a|an|any|the))?"
    r"|(?:(?:don'?t|dont|do\s+not|isn'?t|isnt|aren'?t|arent)\s+"
    r"(?:want|need|like|prefer|care\s+about)(?:\s+(?:a|an|any))?)"
    r")\s*$",
    re.I,
)


def _explicitly_negated(low: str, start: int, window: int = 64) -> bool:
    """Whether the value immediately follows an explicit rejection cue.

    ``_clause_before`` intentionally splits on repair words such as ``instead``.  That is useful
    for ordinary polarity scope, but it also hides the ``instead of X`` rejection from the value
    extractor.  Check the unsplit local prefix first so the negative event wins over any positive
    vocabulary match in the same turn.
    """
    prefix = low[max(0, start - window):start]
    return bool(_EXPLICIT_NEGATION_BEFORE_RE.search(prefix))


def _weight_near(low: str, start: int, window: int = 30) -> float:
    tokens = set(TOKEN_RE.findall(_clause_before(low, start, window)))
    return 0.5 if tokens & SOFT_CUES else 1.0


REPAIR_CUE_RE = re.compile(
    r"\b(?:actually|actly|wait(?:\s+(?:no|nah))?|instead|rather|scratch that|"
    r"never ?mind|changed my mind|make that|make it|i mean|nah|but)\b",
    re.I,
)
_REPAIR_BETWEEN_RE = re.compile(
    r"\b(?:actually|actly|wait(?:\s+(?:no|nah))?|instead|rather|scratch that|"
    r"never ?mind|changed my mind|make that|make it|i mean|nah|no|but)\b",
    re.I,
)
_IMPLICIT_REJECTION_RE = re.compile(
    r"\bwhy\b.{0,80}\b(?:instead|like|give(?:\s+me)?|giving\s+me|"
    r"show(?:ing)?\s+me|recommend(?:ing)?\s+me|get\s+me|want|need)\b",
    re.I,
)
_ADDITIVE_RE = re.compile(r"^\s*(?:,|/|\band\b|\bor\b|\+|-)+\s*$", re.I)
_PREFERENCE_SLOT = {
    "colour": "color", "color": "color", "material": "material", "fabric": "material",
    "size": "size", "style": "style", "brand": "brand", "budget": "budget",
    "category": "category", "feature": "feature", "use case": "use_case",
}
_SLOT_WORDS = "|".join(sorted((re.escape(k) for k in _PREFERENCE_SLOT), key=len, reverse=True))
NO_PREFERENCE_RE = re.compile(
    rf"(?:\b(?:i\s+)?(?:do\s+not|don'?t|dont|no\s+longer)\s+"
    rf"(?:care|mind)(?:\s+about)?\s+(?:the\s+)?(?P<s1>{_SLOT_WORDS})\b"
    rf"|\b(?:i\s+)?(?:do\s+not|don'?t|dont)\s+have\s+"
    rf"(?:(?:a|an|any)\s+)?"
    rf"preference\s+(?:for|on)\s+(?:the\s+)?(?P<s4>{_SLOT_WORDS})\b"
    rf"|\b(?:i\s+)?(?:have\s+)?no\s+preference\s+(?:for|on)\s+(?:the\s+)?"
    rf"(?P<s2>{_SLOT_WORDS})\b"
    rf"|\bno\s+(?P<s5>{_SLOT_WORDS})\s+preference\b"
    rf"|\b(?P<s6>{_SLOT_WORDS})\s+(?:does\s+not|doesn'?t|doesnt)\s+matter\b"
    rf"|\b(?:any|whatever)\s+(?P<s3>{_SLOT_WORDS})\s+(?:is\s+)?"
    rf"(?:fine|okay|ok|works)\b)",
    re.I,
)
CLEAR_PREFERENCE_RE = re.compile(
    rf"\b(?:clear|remove|drop|forget)\s+(?:my\s+|the\s+)?"
    rf"(?P<slot>{_SLOT_WORDS})(?:\s+preference)?\b",
    re.I,
)


class SlotFiller:
    """Parse a message into polarity-aware `Constraint`s. Deterministic, offline, ~O(len)."""

    def __init__(self, vocab: "CatalogVocab | None" = None) -> None:
        self.vocab = vocab

    def parse(self, text: str, turn: int) -> list[Constraint]:
        low = text.lower()
        out: list[Constraint] = []

        def emit(slot: str, value: str, span: tuple[int, int], polarity: int | None = None) -> None:
            # Explicit rejection is authoritative for the value at this span.  This check must
            # happen before the normal local polarity heuristic, whose repair-boundary splitting
            # would otherwise turn ``instead of linen`` into a positive ``linen`` match.
            explicit_negative = _explicitly_negated(low, span[0])
            resolved_polarity = -1 if explicit_negative else (
                _polarity_near(low, span[0]) if polarity is None else polarity)
            out.append(Constraint(slot, _norm(value),
                                  resolved_polarity,
                                  _weight_near(low, span[0]), turn,
                                  span=span, surface=low[span[0]:span[1]]))

        # Boundary and explicit clearing operations are first-class ledger events. Their full
        # spans are masked from the retrieval query, while NO_PREFERENCE also suppresses re-asking.
        for match in NO_PREFERENCE_RE.finditer(low):
            raw_slot = next(v for v in match.groupdict().values() if v is not None)
            out.append(Constraint(
                _PREFERENCE_SLOT[raw_slot], "", 0, 1.0, turn,
                operation="NO_PREFERENCE", span=match.span(), surface=match.group(0),
            ))
        for match in CLEAR_PREFERENCE_RE.finditer(low):
            raw_slot = match.group("slot")
            out.append(Constraint(
                _PREFERENCE_SLOT[raw_slot], "", 0, 1.0, turn,
                operation="CLEAR", span=match.span(), surface=match.group(0),
            ))

        # Feedback can reject the assistant's product type without saying "not":
        # "why loafers ... like running shoe". Treat it as a modifier reset, not as a new
        # loafer preference; the active category/use-case signals in the same message remain.
        for match in _IMPLICIT_REJECTION_RE.finditer(low):
            out.append(Constraint(
                "__modifiers__", "", 0, 1.0, turn,
                operation="CLEAR", span=match.span(), surface=match.group(0),
            ))

        for rx, slot in ((MATERIAL_RE, "material"), (COLOR_RE, "color"),
                         (STYLE_RE, "style"), (SIZE_RE, "size"),
                         (FEATURE_RE, "feature")):
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
            inside_meta = any(
                event.operation in {"CLEAR", "NO_PREFERENCE"} and event.span
                and event.span[0] <= m.start(1) < event.span[1]
                for event in out
            )
            already_extracted = any(
                event.span == m.span(1) and event.value == _norm(w) for event in out
            )
            if (inside_meta or already_extracted or w in STOP_FEATURE
                    or MATERIAL_RE.match(w) or COLOR_RE.match(w) or resolve_category(w)):
                continue
            out.append(Constraint(
                "feature", _norm(w), -1, 1.0, turn, operation="REMOVE",
                span=m.span(1), surface=m.group(1),
            ))

        # Preserve leftmost-head-noun behavior ordinarily. When category mentions are separated
        # by an explicit repair cue, retain them all in span order so the final SET wins.
        category_mentions = _category_matches(low)
        # Users commonly mention the wrong recommendation while correcting the assistant
        # ("why are you giving me snow boots?"). That noun is feedback, not a new preference.
        category_mentions = [
            mention for mention in category_mentions
            if not re.search(
                r"\b(?:why\s+(?:are|do)\s+you\s+|you\s+are\s+|"
                r"giving\s+me\s+|showing\s+me\s+|recommend(?:ing)?\s+me\s+)"
                r"(?:.{0,24})$",
                low[max(0, mention[0] - 48):mention[0]],
                re.I,
            )
        ]
        selected_categories = category_mentions[:1]
        if len(category_mentions) > 1:
            repaired = [category_mentions[0]]
            for previous, current in zip(category_mentions, category_mentions[1:]):
                connector = low[previous[1]:current[0]]
                if _REPAIR_BETWEEN_RE.search(connector):
                    repaired.append(current)
            if len(repaired) > 1:
                selected_categories = repaired
        for start, end, surface, cat in selected_categories:
            emit("category", cat, (start, end), polarity=1)

        # brand via catalog vocab (longest-match), if available
        if self.vocab is not None:
            brand = self.vocab.match_brand(low)
            if brand:
                idx = low.find(brand)
                emit("brand", brand, (idx, idx + len(brand)))

        out.sort(key=lambda event: (
            event.span[0] if event.span else len(low),
            event.span[1] if event.span else len(low),
        ))

        # Convert extracted facts into ordered updates. Exclusive slots SET by default; an
        # explicit conjunction permits ADD. Multi-valued slots ADD unless a repair cue says the
        # new value replaces the earlier one. A repair connector also makes its RHS positive:
        # "red, wait no blue" means SET blue, whereas "not red, blue" means avoid red + want blue.
        previous_by_slot: dict[str, Constraint] = {}
        for event in out:
            if event.operation in {"CLEAR", "NO_PREFERENCE"}:
                previous_by_slot.pop(event.slot, None)
                continue
            previous = previous_by_slot.get(event.slot)
            connector = "" if previous is None or not previous.span or not event.span else (
                low[previous.span[1]:event.span[0]]
            )
            repaired = bool(previous and _REPAIR_BETWEEN_RE.search(connector))
            additive = bool(previous and _ADDITIVE_RE.match(connector))
            no_as_repair = bool(re.search(r"\b(?:wait(?:\s+no)?|no|nah)\b", connector))
            if repaired and no_as_repair and event.polarity < 0 and not re.search(
                    r"\bnot\b", connector):
                event.polarity = 1
            if event.polarity < 0:
                event.operation = "REMOVE"
            elif previous and additive:
                event.operation = "ADD"
            elif event.slot in NeedModel.EXCLUSIVE_SLOTS or repaired:
                event.operation = "SET"
            else:
                event.operation = "ADD"
            previous_by_slot[event.slot] = event

        # A trailing standalone "scratch that" / "never mind" clears the most recent active
        # slot. If a replacement follows the cue, the replacement SET already expresses repair.
        trailing = list(re.finditer(r"\b(?:scratch that|never ?mind)\b", low, re.I))
        if trailing:
            cue = trailing[-1]
            has_later_fact = any(e.span and e.span[0] > cue.end() and e.value for e in out)
            if not has_later_fact:
                out.append(Constraint(
                    "__last__", "", 0, 1.0, turn, operation="CLEAR",
                    span=cue.span(), surface=cue.group(0),
                ))
                out.sort(key=lambda event: event.span[0] if event.span else len(low))

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


class BeliefModel:
    """Turns the ranked pool + scores into an attribute-level belief over the need."""

    TOPN = 20

    def __init__(self, catalog: dict[str, dict], doc_fn, vocab: CatalogVocab) -> None:
        self.catalog = catalog
        self.doc = doc_fn
        self.price_q = vocab.price_quantiles

    def _modal_category(self, head: list[str]) -> str | None:
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
                attr_unc[slot] = max(u, 0.5)      # required-but-unknown stays askable
            else:
                attr_unc[slot] = 1.0

        need_conf = 1.0 - (sum(attr_unc.values()) / len(attr_unc) if attr_unc else 0.0)
        item_conf = 0.5 * margin + 0.3 * (1.0 - ent) + 0.2 * min(stable / 2.0, 1.0)
        conf = min(item_conf, need_conf)
        return Belief(top, margin, ent, stable, cat, item_conf, need_conf, conf, attr_unc)


def converge(belief: Belief, missing: list[str], turn: int, last_turn: int = 10) -> str:
    """Return DELIVER, CONFIRM, or PROBE based on current belief and turn."""
    if belief.confidence >= CONVERGE_HIGH or turn >= last_turn:
        return "DELIVER"
    if belief.item_confidence >= CONVERGE_MID and not missing:
        return "CONFIRM"
    return "PROBE"


# Slot decision weights: how much resolving each slot narrows the candidate pool.
DECISION_WEIGHT = {"budget": 1.3, "size": 1.2, "material": 1.1, "use_case": 1.0,
                   "category": 1.0, "feature": 1.0, "style": 0.9, "color": 0.8}

# Adaptive clarification (QuestionSelector, USE_ADAPTIVE_CLARIFY). Slots whose values `attr_value`
# can extract, so `_top_values(head, slot)` being empty means the pool cannot answer that question.
_EXTRACTED_SLOTS = {"material", "color", "style", "use_case", "size"}
_WORD_RE = re.compile(r"[a-z0-9]+")
# Generic, non-discriminating tokens excluded from the pool-derived feature facet.
_FACET_STOP = {
    "with", "and", "for", "the", "this", "that", "from", "your", "you", "our", "are", "all",
    "womens", "women", "mens", "men", "kids", "girls", "boys", "unisex", "size", "sizes",
    "small", "medium", "large", "pack", "set", "pair", "new", "style", "fashion", "quality",
    "premium", "classic", "made", "design", "designed", "perfect", "great", "features", "product",
    "material", "color", "colors", "available", "please", "will", "can", "has", "have",
}


class QuestionSelector:
    """Component A — ask the question that most reduces the belief's uncertainty (PROBE),
    verifies the top hypothesis (CONFIRM), or steps aside (DELIVER). Pool-aware phrasing."""

    adaptive_clarify = False   # set by Agent from config.USE_ADAPTIVE_CLARIFY

    def __init__(self, catalog: dict[str, dict], doc_fn, price_q: list[float]) -> None:
        self.catalog = catalog
        self.doc = doc_fn
        self.price_q = price_q

    def select(self, belief: Belief, need: NeedModel, conv_state: str,
               head: list[str], guidance: dict[str, float] | None = None) -> tuple[str | None, str]:
        if conv_state == "DELIVER":
            return None, "Here are the closest matches based on what you've told me."
        if conv_state == "CONFIRM" and belief.top_asin:
            attr = self._distinctive_attr(belief.top_asin, head)
            if attr and attr not in need.no_preference:
                return attr, self._confirm_phrase(attr, belief.top_asin)
        # When top candidates are nearly tied, a product comparison question is more
        # discriminating than asking about an abstract attribute.
        if belief.margin < 0.15 and len(head) >= 2:
            cmp = self._comparison_phrase(head)
            if cmp:
                # A comparison prompt still needs a structured evaluator action. Use the most
                # informative unresolved slot as the payload instead of forcing the simulator's
                # generic ``other`` response.
                comparison_attr = self._best_supported_attr(
                    belief.attr_uncertainty, need, guidance)
                if comparison_attr:
                    return comparison_attr, cmp
        # guidance multiplier is per-slot learned info-gain weight (1.0 when unseen)
        unc = dict(belief.attr_uncertainty)
        facet_word: str | None = None
        unc = {s: u for s, u in unc.items() if s not in need.no_preference}
        if self.adaptive_clarify:
            # (a) drop structured slots the candidate pool has no values for — asking them cannot
            # discriminate and just burns a turn (a common leak-free/long-tail failure).
            unc = {s: u for s, u in unc.items()
                   if s not in _EXTRACTED_SLOTS or self._top_values(head, s)}
            # (b) add a pool-derived `feature` facet so feature-classified constraints are askable.
            if not need.has_positive("feature") and "feature" not in need.no_preference:
                facet = self._feature_facet(head, need)
                if facet:
                    facet_word, unc["feature"] = facet[0], facet[1]
        if unc:
            g = guidance or {}
            attr = self._best_supported_attr(unc, need, g)
            if not attr:
                return "other", "Is there a specific detail that matters most to you?"
            if attr == "feature" and facet_word:
                return "feature", f"Any particular feature that matters — like {facet_word}?"
            return attr, self._probe_phrase(attr, head)
        return "other", "Is there a specific detail that matters most to you?"

    @staticmethod
    def _best_supported_attr(
        uncertainty: dict[str, float],
        need: NeedModel,
        guidance: dict[str, float] | None = None,
    ) -> str | None:
        """Choose an unresolved, non-boundary slot suitable for evaluator replies."""
        weights = guidance or {}
        candidates = {
            slot: value for slot, value in uncertainty.items()
            if slot in DECISION_WEIGHT and slot not in need.no_preference
        }
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda slot: (
                candidates[slot] * DECISION_WEIGHT.get(slot, 1.0) * weights.get(slot, 1.0),
                -list(DECISION_WEIGHT).index(slot)
                if slot in DECISION_WEIGHT else 0,
            ),
        )

    def _feature_facet(self, head: list[str], need: NeedModel) -> tuple[str, float] | None:
        """The distinctive token the top candidates most SPLIT on — a facet (waterproof, padded,
        stone type, closure...) that the fixed structured slots don't capture. Pool-derived, so it is
        category-adaptive and needs no hand-maintained map. Returns (token, strength) where strength
        peaks at a 50/50 split (maximal information gain). None if nothing splits the pool."""
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
            if c < 2 or c > n - 2:                 # present in ~half → discriminating
                continue
            frac = c / n
            strength = 1.0 - abs(0.5 - frac) * 2.0   # 1.0 at 50/50, 0 at all/none
            if strength >= 0.5 and (best is None or strength > best[1]):
                best = (tok, strength)
        return best

    # ---- phrasing (fills specifics from the actual candidate head) ----
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
        for slot in ("material", "color", "style", "size", "use_case", "budget", "category"):
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
        """When top candidates are nearly tied, ask a direct product comparison question.

        More discriminating than abstract attribute questions when the top-2 differ on
        multiple dimensions simultaneously — the shopper names their preference directly.
        """
        a1, a2 = head[0], head[1]
        t1 = str(self.catalog.get(a1, {}).get("title") or "")
        t2 = str(self.catalog.get(a2, {}).get("title") or "")
        if not t1 or not t2 or t1[:40] == t2[:40]:
            return None
        p1 = self._display_price(self.catalog.get(a1, {}).get("price"))
        p2 = self._display_price(self.catalog.get(a2, {}).get("price"))
        desc1 = f"{t1[:50]}" + (f" (${p1:.0f})" if p1 is not None else "")
        desc2 = f"{t2[:50]}" + (f" (${p2:.0f})" if p2 is not None else "")
        return f"Are you looking for something more like '{desc1}' or '{desc2}'?"

    @staticmethod
    def _display_price(value: object) -> float | None:
        """Normalize heterogeneous catalog prices for comparison prompts.

        Some frozen catalog rows contain numeric strings rather than JSON numbers.  Ranking has
        already completed when this prompt is built, so allowing a formatting exception here makes
        the evaluator discard an otherwise valid top-10 response and falsely report a rank miss.
        Missing, malformed, and non-positive prices are simply omitted from the prompt.
        """
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return price if price > 0 else None


class RationaleBuilder:
    """Builds a short "why this matches" string from constraints the top candidate satisfies."""

    def __init__(self, catalog: dict[str, dict], doc_fn) -> None:
        self.catalog = catalog
        self.doc = doc_fn

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
                continue                                   # implied by the result itself
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


def apply_negatives(candidates: list[str], need: NeedModel, doc_fn) -> list[str]:
    """Demote avoid-constraint violators to the back without dropping any candidates."""
    negs = [c.value for c in need.negatives() if c.value]
    if not negs or not candidates:
        return candidates
    keep, drop = [], []
    for asin in candidates:
        text = doc_fn(asin)
        (drop if any(v in text for v in negs) else keep).append(asin)
    return keep + drop
