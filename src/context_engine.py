"""Context engine: session distillation, user profiles, orchestration policy, guidance learning.

═══ OPTIONAL LAYER — on by default but UNPROVEN on the scored path. ═══
This is the Dynamic Context Programming (self-evolution / short + long-term memory) capability. Its
defaults reproduce the static pipeline exactly, so it is score-NEUTRAL, not negative; long-term
profiles are dormant in evaluation because public/private sessions are distinct users. It is a
product-facing capability pending an ablation (WS4) to show it moves the metric. See src/agent.py
flag ledger (USE_DCP family).

Four components:
  ContextDistiller    — recency decay, volatility tracking, salience pruning per turn
  ProfileService      — persistent per-user preferences with time-decay
  OrchestrationPolicy — per-turn execution plan (routes, weights, pool size)
  GuidanceLearner     — online reweighting of clarification questions by realized info-gain

All four are benchmark-neutral: in offline evaluation each session resets independently,
so profile write-through never affects a subsequent session's read.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.understanding import Belief, Constraint, NeedModel


@dataclass
class SessionContext:
    """Compact, weighted, decayed view of the running dialog. Bounded by salience pruning."""

    need: NeedModel
    belief: Belief
    intent_trace: list[float] = field(default_factory=list)   # buying_score history
    volatility: float = 0.0                                    # constraint churn t-1 -> t
    turn: int = 0
    _prev_keys: frozenset = field(default_factory=frozenset)   # for churn detection

    def snapshot(self) -> dict:
        """Compact, loggable (~1KB) view for :state / structured logs."""
        return {
            "turn": self.turn,
            "volatility": round(self.volatility, 3),
            "intent_trace": [round(x, 2) for x in self.intent_trace[-6:]],
            "constraints": [
                {"slot": c.slot, "value": c.value, "pol": c.polarity, "w": round(c.weight, 2)}
                for c in self.need.constraints
            ],
            "category": self.need.category,
            "confidence": round(self.belief.confidence, 3),
            "conv": getattr(self.belief, "category", None),
        }


class ContextDistiller:
    """Recency-decays stale constraints, tracks volatility, and salience-prunes to bound context.

    Weights/pruning only affect soft signals; the coverage reranker uses verbatim
    constraint_phrases separately, so ranking is unaffected.
    """

    DECAY = 0.9
    MAX_CONSTRAINTS = 12
    MIN_KEEP = 4          # floor to avoid over-pruning on high-override sessions
    PRUNE_FLOOR = 0.15

    def update(self, ctx: SessionContext | None, need: NeedModel, belief: Belief,
               buying_score: float, turn: int) -> SessionContext:
        if ctx is None:
            ctx = SessionContext(need=need, belief=belief)
        fresh = frozenset(c.key() for c in need.constraints if c.turn == turn)
        prev = ctx._prev_keys
        all_keys = frozenset(c.key() for c in need.constraints)

        union = prev | all_keys
        ctx.volatility = 1.0 - (len(prev & all_keys) / len(union)) if union else 0.0

        for c in need.constraints:
            if c.key() not in fresh and c.turn < turn:
                c.weight *= self.DECAY

        self._prune(need)

        ctx.need = need
        ctx.belief = belief
        ctx.intent_trace.append(buying_score)
        ctx.turn = turn
        ctx._prev_keys = all_keys
        return ctx

    def _prune(self, need: NeedModel) -> None:
        if len(need.constraints) <= max(self.MIN_KEEP, 0):
            return
        prunable = [c for c in need.constraints
                    if c.slot != "category" and c.weight < self.PRUNE_FLOOR]
        if not prunable:
            if len(need.constraints) <= self.MAX_CONSTRAINTS:
                return
            prunable = sorted(
                [c for c in need.constraints if c.slot != "category"],
                key=lambda c: c.weight)
        keep = need.constraints
        drop = set()
        for c in sorted(prunable, key=lambda c: c.weight):
            if len(keep) - len(drop) <= self.MIN_KEEP:
                break
            if len(keep) - len(drop) <= self.MAX_CONSTRAINTS and c.weight >= self.PRUNE_FLOOR:
                break
            drop.add(id(c))
        if drop:
            need.constraints = [c for c in need.constraints if id(c) not in drop]


@dataclass
class ProfilePreference:
    slot: str
    value: str
    weight: float
    last_seen_ts: float

    def as_dict(self) -> dict:
        return {"slot": self.slot, "value": self.value,
                "weight": round(self.weight, 4), "last_seen_ts": self.last_seen_ts}


@dataclass
class UserProfile:
    user_id: str
    prefs: list[ProfilePreference] = field(default_factory=list)
    category_affinity: dict[str, float] = field(default_factory=dict)
    guidance_bias: dict[str, float] = field(default_factory=dict)
    schema_v: int = 1

    def preference_tags(self) -> list[str]:
        """Durable positive values, strongest first — seeds the Personalizer at warm-start."""
        return [p.value for p in sorted(self.prefs, key=lambda p: -p.weight)]

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id, "schema_v": self.schema_v,
            "prefs": [p.as_dict() for p in self.prefs],
            "category_affinity": {k: round(v, 4) for k, v in self.category_affinity.items()},
            "guidance_bias": {k: round(v, 4) for k, v in self.guidance_bias.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        return cls(
            user_id=d.get("user_id", ""),
            prefs=[ProfilePreference(**p) for p in d.get("prefs", [])],
            category_affinity=dict(d.get("category_affinity", {})),
            guidance_bias=dict(d.get("guidance_bias", {})),
            schema_v=d.get("schema_v", 1),
        )


class ProfileService:
    """Persistent, time-decaying per-user preference store (cache/profiles.json).

    Keyed by a hash of the anonymized profile; in offline evaluation each session is fresh,
    so write-through never feeds a subsequent session's read.
    """

    EMA = 0.6              # write-through blend of durable weight toward the new session
    HALFLIFE_DAYS = 45.0   # decay applied at read time
    PRUNE_EPS = 0.05

    def __init__(self, path: str | None = None) -> None:
        from src.config import PROFILE_STORE
        path = path or PROFILE_STORE
        self.path = Path(path)
        self._store: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._store = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._store = {}

    @staticmethod
    def user_id(profile: dict) -> str:
        """Opaque, stable id from the provided profile (no PII stored)."""
        if profile.get("user_id"):
            return str(profile["user_id"])
        basis = json.dumps(
            {"t": sorted(profile.get("preference_tags") or []),
             "s": profile.get("summary") or ""}, sort_keys=True)
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]

    def load(self, profile: dict) -> UserProfile:
        uid = self.user_id(profile)
        raw = self._store.get(uid)
        if raw:
            up = UserProfile.from_dict(raw)
            self._decay(up)
            return up
        prefs = [ProfilePreference("tag", t, 1.0, time.time())
                 for t in (profile.get("preference_tags") or []) if t]
        return UserProfile(user_id=uid, prefs=prefs)

    def _decay(self, up: UserProfile) -> None:
        now = time.time()
        kept: list[ProfilePreference] = []
        for p in up.prefs:
            age_days = max(0.0, (now - p.last_seen_ts) / 86400.0)
            p.weight *= 0.5 ** (age_days / self.HALFLIFE_DAYS)
            if p.weight >= self.PRUNE_EPS:
                kept.append(p)
        up.prefs = kept

    def write_through(self, up: UserProfile, ctx: SessionContext) -> None:
        """Merge the distilled session's positive constraints into durable prefs (EMA +
        recency), update category affinity, then persist (best-effort)."""
        now = time.time()
        index = {(p.slot, p.value): p for p in up.prefs}
        for c in ctx.need.positives():
            if c.slot == "category":
                up.category_affinity[c.value] = min(
                    1.0, up.category_affinity.get(c.value, 0.0) + 0.25 * c.weight)
            key = (c.slot, c.value)
            if key in index:
                p = index[key]
                p.weight = self.EMA * c.weight + (1 - self.EMA) * p.weight
                p.last_seen_ts = now
            else:
                p = ProfilePreference(c.slot, c.value, c.weight, now)
                up.prefs.append(p)
                index[key] = p
        self._store[up.user_id] = up.as_dict()
        self._flush()

    def merge_guidance(self, up: UserProfile, slot: str, gain: float, lam: float = 0.3) -> None:
        prev = up.guidance_bias.get(slot, 0.0)
        up.guidance_bias[slot] = (1 - lam) * prev + lam * gain
        self._store[up.user_id] = up.as_dict()

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._store), encoding="utf-8")
        except Exception:
            pass  # best-effort; failure here must not affect a scoring run


@dataclass
class ExecutionPlan:
    """Per-turn pipeline configuration emitted by OrchestrationPolicy."""
    routes: list[str]
    route_weights: dict[str, float]
    pool_size: int
    rerank_stack: list[str]  # coverage must be last
    dialogue_action: str     # "PROBE" | "CONFIRM" | "DELIVER"
    ask_slot: str | None = None
    rationale: bool = True

    def as_dict(self) -> dict:
        return {
            "routes": self.routes, "route_weights": {k: round(v, 3) for k, v in self.route_weights.items()},
            "pool_size": self.pool_size, "rerank_stack": self.rerank_stack,
            "dialogue_action": self.dialogue_action, "ask_slot": self.ask_slot,
        }


class OrchestrationPolicy:
    """Emits a per-turn ExecutionPlan: routes, weights, pool size, and rerank order.

    Default values reproduce the static pipeline exactly (same interpolation, pool=200,
    personalization then coverage-last), so enabling this is score-neutral.
    """

    DENSE_BUYING = 0.20
    DENSE_BROWSING = 0.35
    BASE_POOL = 200
    EXPANSION_WEIGHT = 0.10

    def plan(self, ctx: SessionContext, buying_score: float, conv_state: str,
             ask_slot: str | None, warm: bool) -> ExecutionPlan:
        dense = self.DENSE_BROWSING + buying_score * (self.DENSE_BUYING - self.DENSE_BROWSING)
        weights = {"dense": dense, "expansion": self.EXPANSION_WEIGHT}
        return ExecutionPlan(
            routes=["bm25", "dense", "expansion"],
            route_weights=weights,
            pool_size=self.BASE_POOL,
            rerank_stack=["personalization", "coverage"],
            dialogue_action=conv_state,
            ask_slot=ask_slot,
        )


class GuidanceLearner:
    """Online reweighting of clarification question slots by realized information gain.

    Measures belief entropy drop after each question and adjusts future slot priorities.
    Cold-start-safe: no stats → multiplier = 1.0 → falls back to static DECISION_WEIGHT.
    """

    LAMBDA = 0.5  # guidance multiplier strength
    EMA = 0.3     # online update rate

    def __init__(self, path: str | None = None) -> None:
        from src.config import GUIDANCE_STORE
        path = path or GUIDANCE_STORE
        self.path = Path(path)
        self.stats: dict[str, float] = {}
        self.waveoff: dict[str, float] = {}
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.stats = dict(d.get("stats", {}))
                self.waveoff = dict(d.get("waveoff", {}))
            except Exception:
                pass

    def observe(self, prev_ask: str | None, prev_entropy: float, prev_conf: float,
                belief: Belief, waved_off: bool, up: UserProfile | None,
                svc: "ProfileService | None") -> float:
        """Called at turn t+1 about the question asked at t. Returns the realized gain."""
        if not prev_ask or prev_ask == "other":
            return 0.0
        gain = max(0.0, prev_entropy - belief.entropy) + max(0.0, belief.confidence - prev_conf)
        # update global
        self.stats[prev_ask] = (1 - self.EMA) * self.stats.get(prev_ask, 0.0) + self.EMA * gain
        wo = 1.0 if waved_off else 0.0
        self.waveoff[prev_ask] = (1 - self.EMA) * self.waveoff.get(prev_ask, 0.0) + self.EMA * wo
        self._flush()
        # update per-user
        if up is not None and svc is not None:
            svc.merge_guidance(up, prev_ask, gain, self.LAMBDA)
        return gain

    def weights(self, up: UserProfile | None) -> dict[str, float]:
        """Multiplier per slot for the QuestionSelector: (1 + λ·gain)·(1 − waveoff), blended
        with any per-user bias. 1.0 when unseen (falls back to static DECISION_WEIGHT)."""
        out: dict[str, float] = {}
        slots = set(self.stats) | set(self.waveoff)
        if up is not None:
            slots |= set(up.guidance_bias)
        for s in slots:
            g = self.stats.get(s, 0.0)
            if up is not None and s in up.guidance_bias:
                g = 0.5 * g + 0.5 * up.guidance_bias[s]
            mult = (1.0 + self.LAMBDA * g) * (1.0 - min(0.9, self.waveoff.get(s, 0.0)))
            out[s] = mult
        return out

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"stats": self.stats, "waveoff": self.waveoff}), encoding="utf-8")
        except Exception:
            pass
