"""Intent routing, per-session conversation state, and dialogue decision logic."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from src.config import (
    ASK_PRIORITY, CONFIDENCE_EMA, DELIVER_TURN_THRESHOLD, EXPLORE_TERM_THRESHOLD,
    INTENT_BROWSING_CUE_WEIGHT, INTENT_BROWSING_CUTOFF, INTENT_BUYING_CUE_WEIGHT,
    INTENT_BUYING_CUTOFF, INTENT_HARD_CONSTRAINT_WEIGHT, INTENT_SPECIFICITY_PIVOT,
    INTENT_SPECIFICITY_SLOPE,
)
from src.understanding import Belief, NeedModel

# ---------------------------------------------------------------------------
# Constraint marker regex — the simulator signals a hard constraint with these phrases.
# Text after the marker is lifted verbatim from the target's catalog fields.
_CONSTRAINT_MARKER_RE = re.compile(
    r"(?:key requirement is|what matters is|what i need is)\s*:\s*(.+)",
    re.I,
)

# Hard-constraint signal words used by the intent router
_HARD_CONSTRAINT_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|alloy|"
    r"stainless|steel|silver|gold|black|white|blue|red|pink|green|brown|gray|grey|"
    r"purple|yellow|orange|size|\$\d|under\s*\$?\d|waterproof|resistant)\b",
    re.I,
)


def extract_constraints(message: str) -> list[str]:
    """Return verbatim constraint phrases from a simulator turn (split on ';').

    Do not normalise — phrases are matched verbatim by CoverageReranker.
    """
    m = _CONSTRAINT_MARKER_RE.search(message)
    if not m:
        return []
    tail = m.group(1).strip().rstrip(".")
    return [part.strip() for part in tail.split(";") if len(part.strip()) > 2]


class IntentRouter:
    """Produces a continuous buying_score ∈ [0, 1] for smooth retrieval weight blending."""

    OVERRIDE = ("actually, ignore", "ignore my earlier", "never mind my earlier")
    BUYING = ("key requirement", "must have", "need exactly", "a key requirement is")
    BROWSING = ("still exploring", "not sure", "just browsing", "exploring", "ideas")

    def is_override(self, message: str) -> bool:
        low = message.lower()
        return any(t in low for t in self.OVERRIDE)

    def score(self, message: str, distinct_terms: int) -> float:
        """Buying-ness in [0, 1]: 1 = precise/high-intent, 0 = open browsing."""
        low = message.lower()
        s = 0.0
        if any(t in low for t in self.BUYING):
            s += INTENT_BUYING_CUE_WEIGHT
        if any(t in low for t in self.BROWSING):
            s -= INTENT_BROWSING_CUE_WEIGHT
        if _HARD_CONSTRAINT_RE.search(low):
            s += INTENT_HARD_CONSTRAINT_WEIGHT
        # specificity: few distinct terms → browsing, many → buying
        s += INTENT_SPECIFICITY_SLOPE * (distinct_terms - INTENT_SPECIFICITY_PIVOT)
        return 1.0 / (1.0 + math.exp(-s))

    @staticmethod
    def label(buying_score: float) -> str:
        if buying_score >= INTENT_BUYING_CUTOFF:
            return "buying"
        if buying_score <= INTENT_BROWSING_CUTOFF:
            return "browsing"
        return "mixed"


# ---------------------------------------------------------------------------
@dataclass
class ConversationState:
    """All per-session state. One instance per session; reset() creates a fresh one.

    Fields are grouped by concern:
      Core session:  user_profile, all_text, asked_attrs, boundary_attrs,
                     intent, buying_score, phase, last_pool, constraint_phrases
      NLU layer:     need, belief, conv_state, ig_attr, ig_phrasing,
                     prev_ask, prev_entropy, prev_conf
      DCP:           ctx, profile, plan
    """
    # ---- Core session ----
    user_profile: dict
    all_text: list[str] = field(default_factory=list)
    asked_attrs: set = field(default_factory=set)
    boundary_attrs: set = field(default_factory=set)
    intent: str = "unknown"
    buying_score: float = 0.5
    phase: str = "explore"
    last_pool: int = 0
    constraint_phrases: list = field(default_factory=list)
    override_turn: int | None = None  # turn number of last intent override (for pool boost)

    need: NeedModel = field(default_factory=NeedModel)
    belief: Belief = field(default_factory=Belief)
    conv_state: str = "PROBE"
    ig_attr: str | None = None
    ig_phrasing: str | None = None
    prev_ask: str | None = None
    prev_entropy: float = 0.0
    prev_conf: float = 0.0

    ctx: object = None      # SessionContext | None
    profile: object = None  # UserProfile | None
    plan: object = None     # ExecutionPlan | None

    def accumulate(self, message: str) -> None:
        self.all_text.append(message)

    def query_text(self) -> str:
        return " ".join(self.all_text)


def phase_transition(
    state: ConversationState, turn: int, pool_size: int
) -> str:
    """Return the dialogue phase for this turn: explore, converge, or deliver."""
    from src.catalog import terms  # avoid circular at module level
    distinct = len(set(terms(state.query_text())))
    if distinct < EXPLORE_TERM_THRESHOLD and state.last_pool >= pool_size:
        return "explore"
    if turn >= DELIVER_TURN_THRESHOLD:
        return "deliver"
    return "converge"


def next_ask(state: ConversationState, use_info_gain: bool, info_gain_mode: str) -> str | None:
    """Return the attribute to request next.

    In 'display' mode (benchmark-safe): always ask 'other' to maximize constraint
    extraction; info-gain phrasing is voiced in the message but doesn't change ask_attribute.
    In 'ask' mode: the info-gain selector drives the actual ask_attribute field.
    """
    if use_info_gain and info_gain_mode == "ask":
        return state.ig_attr

    for attr in ASK_PRIORITY:
        if attr in state.boundary_attrs:
            continue
        if attr == "other":
            return attr
        if attr not in state.asked_attrs:
            state.asked_attrs.add(attr)
            return attr
    return "other"


_DEFAULT_PROMPTS: dict[str, str] = {
    "other": "Is there a specific detail that matters most to you?",
    "feature": "Any particular feature you need?",
    "material": "Do you have a material preference?",
    "color": "Any color preference?",
    "style": "What style are you after?",
    "size": "What size do you need?",
    "use_case": "What will you mainly use it for?",
}


def compose_message(
    ask_attr: str | None,
    state: ConversationState,
    use_info_gain: bool,
) -> str:
    """Return the agent's reply message.

    Priority: belief-driven info-gain phrasing (if enabled) > intent-aware
    phrasing > default attribute prompts.
    """
    if use_info_gain and state.ig_phrasing:
        return state.ig_phrasing
    if not ask_attr:
        return "Here are the closest matches I found."
    if state.intent == "browsing" and state.phase == "explore":
        return "To narrow things down, what matters most to you about this item?"
    return _DEFAULT_PROMPTS.get(ask_attr, "Could you tell me more about what you want?")
