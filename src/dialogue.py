"""Intent routing, per-session conversation state, and dialogue decision logic."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from src.catalog import terms
from src.config import (
    ASK_PRIORITY, CONFIDENCE_EMA, DELIVER_TURN_THRESHOLD, EXPLORE_TERM_THRESHOLD,
)
from src.understanding import Belief, NeedModel, REPAIR_CUE_RE

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

_QUERY_FILLER_RE = re.compile(
    r"\b(?:oh|hmm+|hm+|uh+|um+|eh|well|okay|ok|actly|actually|wait|nah|"
    r"instead|rather|scratch that|never ?mind|changed my mind|make that|"
    r"i mean|fuck|fucking|lets|let's|not|no|and|but|remove|drop|ditch|skip|"
    r"exclude|better|gimme|gimmie|like|kind|bro|why|giving|make|shit|"
    r"something|anything|specific|particular|detail|details|most)\b",
    re.I,
)


def _remove_term(text: str, term: str) -> str:
    if not term:
        return text
    return re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", " ", text, flags=re.I)


def _clean_query_text(text: str) -> str:
    text = REPAIR_CUE_RE.sub(" ", text)
    text = _QUERY_FILLER_RE.sub(" ", text)
    text = re.sub(r"[^\w$-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
            s += 1.5
        if any(t in low for t in self.BROWSING):
            s -= 1.5
        if _HARD_CONSTRAINT_RE.search(low):
            s += 1.0
        s += 0.18 * (distinct_terms - 6)  # specificity: few terms→browsing, many→buying
        return 1.0 / (1.0 + math.exp(-s))

    @staticmethod
    def label(buying_score: float) -> str:
        if buying_score >= 0.6:
            return "buying"
        if buying_score <= 0.4:
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
    message_turns: list[int] = field(default_factory=list)
    # Set by the retrieval layer once a disclosed phrase has exact catalog evidence.  A leaky
    # session intentionally follows the historical raw-transcript retrieval path; ordinary and
    # paraphrased sessions keep the correction-safe ledger projection below.
    leaky_evidence: bool = False
    # Stronger ranking-only signal. A one-phrase match can justify a raw retrieval probe but is
    # too common in paraphrased catalogs to safely enable the high popularity prior.
    leaky_ranking_evidence: bool = False
    asked_attrs: set = field(default_factory=set)
    boundary_attrs: set = field(default_factory=set)
    category_anchor: str | None = None
    intent: str = "unknown"
    buying_score: float = 0.5
    phase: str = "explore"
    last_pool: int = 0
    constraint_phrases: list = field(default_factory=list)
    # Parallel turn metadata lets the ledger retire phrase-level simulator disclosures when a
    # repair/category switch supersedes them.  The list is optional for backwards compatibility
    # with tests/callers that assign ``constraint_phrases`` directly.
    constraint_phrase_turns: list[int] = field(default_factory=list)

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

    def accumulate(self, message: str, turn: int | None = None) -> None:
        self.all_text.append(message)
        self.message_turns.append(turn if turn is not None else len(self.all_text))

    def query_text(self) -> str:
        """Return the active retrieval projection, or raw history for detected catalog leaks.

        Raw transcript text remains in ``all_text`` for audit and optional response generation,
        but it is never a retrieval input on the honest path. Negative values are handled by
        ``apply_negatives`` and must not become positive BM25/dense terms there.
        """
        if self.leaky_evidence:
            # This is deliberately the origin/main compatibility path.  It is enabled only after
            # catalog-backed exact evidence is observed by Agent._retrieve; it must not be the
            # default because abandoned corrections would otherwise re-enter honest queries.
            return " ".join(self.all_text)

        values: list[str] = []
        seen: set[str] = set()
        # Ledger order gives stable recency semantics while NeedModel.constraints guarantees that
        # superseded events are inactive. Include canonical category explicitly as the anchor.
        for event in self.need.ledger:
            if event.active and event.polarity > 0 and event.value:
                value = event.value.strip()
                key = value.casefold()
                if key not in seen:
                    seen.add(key)
                    values.append(value)
        if self.need.category:
            category_key = self.need.category.casefold()
            values = [value for value in values if value.casefold() != category_key]
            values.insert(0, self.need.category)
        if self.category_anchor:
            same_as_category = bool(
                self.need.category
                and self.category_anchor.casefold() == self.need.category.casefold()
            )
            if not same_as_category:
                values.insert(1 if values else 0, self.category_anchor)
        # Evaluator-disclosed phrases are active natural-language constraints, not raw chat
        # history. Include their sanitized projection so uncommon catalog terms remain searchable.
        phrases = self.effective_constraint_phrases()
        fallback = self._unparsed_active_terms(values, phrases)
        return " ".join(values + [phrase for phrase in phrases if phrase] + fallback)

    def _unparsed_active_terms(self, active_values: list[str], phrases: list[str]) -> list[str]:
        """Project useful words from the latest active sentence that regex slots missed.

        The structured ledger remains authoritative.  This is only a retrieval recall side-track
        for natural descriptions such as ``rich napped pile``.  We inspect the latest turn (never
        the whole transcript), keep the suffix after the last repair cue, strip evaluator marker
        payloads already represented by ``constraint_phrases``, and remove both retired and already
        structured terms.  ``catalog.terms`` also drops apostrophe fragments and conversational
        stopwords, preventing ``I'm``/``Valentine's`` from re-entering the query.
        """
        if not self.all_text:
            return []
        text = self.all_text[-1]
        marker = _CONSTRAINT_MARKER_RE.search(text)
        if marker:
            # The marker payload is already represented as a phrase and should not be duplicated
            # as an unparsed raw sentence.
            text = text[:marker.start()]
        repairs = list(REPAIR_CUE_RE.finditer(text))
        if repairs:
            text = text[repairs[-1].end():]

        # Remove active structured values and all retired terms before tokenisation.  Surface forms
        # are preferred because they include multi-word values and preserve word boundaries.
        for event in self.need.ledger:
            if event.turn == (self.message_turns[-1] if self.message_turns else len(self.all_text)):
                if event.surface:
                    text = _remove_term(text, event.surface)
                elif event.value:
                    text = _remove_term(text, event.value)
        for term in self.need.excluded_terms():
            text = _remove_term(text, term)
        for value in [*active_values, *phrases]:
            for token in terms(value):
                text = _remove_term(text, token)

        text = _clean_query_text(text)
        active_tokens = set(terms(" ".join(active_values + phrases)))
        excluded_tokens = set(terms(" ".join(self.need.excluded_terms())))
        generic_tokens = {
            "something", "anything", "specific", "particular", "detail", "details", "most",
            "thing", "things", "item", "items", "kind", "sort", "type",
        }
        return [token for token in terms(text)
                if token not in active_tokens
                and token not in excluded_tokens
                and token not in generic_tokens]

    def effective_constraint_phrases(self) -> list[str]:
        """Constraint-marker phrases with rejected/superseded values strictly removed."""
        excluded = self.need.excluded_terms()
        result: list[str] = []
        turns = self.constraint_phrase_turns
        # A direct assignment by an older caller has no metadata; treat those phrases as current.
        if len(turns) != len(self.constraint_phrases):
            turns = [0] * len(self.constraint_phrases)
        for phrase, _turn in zip(self.constraint_phrases, turns):
            cleaned = phrase
            for term in excluded:
                cleaned = _remove_term(cleaned, term)
            cleaned = _clean_query_text(cleaned)
            if cleaned:
                result.append(cleaned)
        return result

    def invalidate_historical_phrases(self, turn: int) -> None:
        """Retire simulator phrase disclosures from turns before a correction.

        Phrase strings are intentionally kept separate from structured constraints for coverage
        ranking, so a ledger CLEAR/SET cannot otherwise deactivate them.  On a category switch or
        explicit repair, retain only disclosures made on the replacement turn.
        """
        if len(self.constraint_phrase_turns) != len(self.constraint_phrases):
            self.constraint_phrase_turns = [turn] * len(self.constraint_phrases)
        kept = [
            (phrase, phrase_turn)
            for phrase, phrase_turn in zip(self.constraint_phrases, self.constraint_phrase_turns)
            if phrase_turn >= turn
        ]
        self.constraint_phrases = [phrase for phrase, _ in kept]
        self.constraint_phrase_turns = [phrase_turn for _, phrase_turn in kept]


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
