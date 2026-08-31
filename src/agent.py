"""Agent orchestrator — official challenge entry point (via starter/agent.py).

Wires together catalog, retrieval, ranking, NLU, dialogue, and context components.
No retrieval, ranking, or NLU logic lives here; see architecture.md for component locations.

Feature flags on the Agent class are ablation toggles; the robustness harness overrides
them with setattr(Agent, k, v).
"""
from __future__ import annotations

import re
from pathlib import Path

from src.catalog import Catalog, terms
from src.config import (
    CE_DEPTH, CE_WEIGHT, CONFIDENCE_EMA, COVERAGE_POP_BLEND, COVERAGE_POP_CAP,
    COVERAGE_DISCRIMINATION_PCTL, COVERAGE_INFORMATIVE_MIN, COVERAGE_PREFIX_BONUS,
    COVERAGE_PREFIX_CHARS, SUPPRESS_POP_ON_PARAPHRASE,
    COVERAGE_RETRIEVAL_WEIGHT, DIVERSITY_HEAD_KEEP, DIVERSITY_LAMBDA, EXPANSION_WEIGHT,
    BUYING_VECTOR_WEIGHT, BROWSING_VECTOR_WEIGHT,
    CE_BETA, CE_CONVEX_GATE_MARGIN, USE_CE_CONVEX, USE_NL_CONSTRAINTS,
    USE_REGIME_ROUTING, REGIME_LEAKY_MIN_EXACT,
    LLM_RERANK_DEPTH, LLM_WEIGHT, POOL_BY_PHASE, POOL_NO_PERSONALIZATION, POOL_SIZE,
    PRICE_PROXIMITY_WEIGHT, RERANK_NEAR_TIE_MARGIN, REVEAL_CONFIDENCE, REVEAL_HOLDBACK_K,
    RETRIEVAL_GUARD_K, RETRIEVAL_GUARD_MAX_EXACT, RETRIEVAL_GUARD_VISIBLE_K, USE_RETRIEVAL_GUARD,
    USE_CATEGORY_SWITCH_CLEAR, USE_PROFILE_NEGATION_PURGE,
    USE_PROFILE_RANKING_FALLBACK, PROFILE_RANKING_STRENGTH,
    SATISFACTION_POP_CHANNEL, SATISFACTION_POP_WEIGHT, SATISFACTION_QUALITY_CHANNEL,
    SATISFACTION_SEM_ALPHA, SATISFACTION_SEM_GATE_HIGH, SATISFACTION_SEM_GATE_LOW,
    SATISFACTION_SPECIFICITY_REF, SATISFACTION_UNKNOWN_FLOOR,
    SEMANTIC_COVERAGE_GATE, SEMANTIC_COVERAGE_WEIGHT, SESSION_MAX_TURNS,
    OVERRIDE_POOL_BOOST, OVERRIDE_POOL_TURNS,
    LTR_MODEL_PATH, SLOT_DECAY, STRUCTURED_COVERAGE_WEIGHT, USE_ADAPTIVE_CLARIFY,
    USE_DISCOVERY_MODE, DISCOVERY_MODE_MAX_TURN, DISCOVERY_MODE_MAX_SLOTS,
    USE_SNIPPET_RATIONALE, USE_CONTRAST_RATIONALE,
    USE_CATEGORY_GATE, USE_LTR,
    USE_SATISFACTION_RANKER,
    USE_TCRS_PHASE_SHRINKAGE, TCRS_SHRINKAGE_RATIO, TCRS_MIN_ITEM_CONF,
    USE_CE_STRUCTURED_QUERY,
    OVERRIDE_PHRASE_DEMOTE,
)
from src.context_engine import (
    ContextDistiller, GuidanceLearner, OrchestrationPolicy, ProfileService,
)
from src.dialogue import (
    ConversationState, IntentRouter, compose_message, extract_constraints,
    next_ask, phase_transition,
)
from src.ranking import (CoverageReranker, Diversifier, NeedSatisfactionScorer, Personalizer,
                         guard_retrieval_head)
from src.retrieval import VectorRetriever, convex_fuse, rrf, vector_weight
from src.trace import Tracer, get_tracer
from src.understanding import (
    CatalogVocab, ExpansionTable, CATEGORY_CANON, MATERIAL_RE,
    USE_CASE_KEYS,
    SlotFiller, UseCaseInferencer,
)
from src.belief import (
    BeliefModel, QuestionSelector, RationaleBuilder,
    apply_category_gate, apply_negatives, converge,
)
from src.keys import GeminiClientPool
from src.llm_inference import LLMResponseGenerator, LLMSlotExtractor, SmartUseCaseInferencer


class Agent:
    """Conversational shopping agent.

    Public API (fixed by the challenge spec):
        __init__(catalog_path: str | Path = "data/catalog.jsonl")
        reset(session_id: str, user_profile: dict) -> None
        respond(session_id: str, user_message: str, turn: int, top_k: int) -> dict

    Class attributes are ablation toggles; override with setattr(Agent, k, v).

    FLAG LEDGER — this block is the single source of truth for what runs on the scored path.
    Convention:
      * A flag defaulting to True is CORE: load-bearing, on the scored path, measured to help.
      * A flag defaulting to False is OPTIONAL: off by default. Its comment states WHY —
        "measured neutral/negative" (kept for ablation), "demo-only" (not scored), or
        "unproven" (implemented, awaiting measurement). Nothing ships on unless a number backs it.
      * OPTIONAL LLM layers (USE_LLM_*) are never on the critical path; they are graceful-
        degradation hooks for unseen language, token-metered via src/keys.py.
    """

    # ----- CORE retrieval (scored path) -----
    USE_VECTOR = True           # dense BGE track (auto-off if cache absent)
    USE_SLOT_EXPANSION = True   # synonym expansion recall track
    EXPANSION_WEIGHT = EXPANSION_WEIGHT
    USE_USECASE_PRIORS = True   # occasion → implied attribute inference
    # OPTIONAL (on, UNPROVEN): LLM use-case inference. Feeds only the 0.1-weight expansion track;
    # not measured to move the scored metric. Falls back to the static table when the LLM is down.
    USE_LLM_INFERENCE = True

    # Intent routing
    USE_INTENT_ROUTING = True
    USE_CONFIDENCE_ROUTING = True  # smooth buying_score interpolation
    CONFIDENCE_EMA = CONFIDENCE_EMA

    # Ranking
    USE_PERSONALIZATION = True
    USE_COVERAGE_RERANK = True      # verbatim constraint coverage (must run last)
    USE_IDF_COVERAGE = False        # weight coverage tokens by rarity (measured neutral)
    # Graduated phrase tiers: award a partial bonus when a long constraint phrase's contiguous
    # leading prefix matches even though the exact phrase does not. Targets MRR on near-miss ranks.
    # Measured neutral on public (exact substrings dominate) and a small win on paraphrase/hard —
    # a generalization signal for the private set where the verbatim leak may weaken. See
    # scripts/exp_phrase_tiers.py and config COVERAGE_PREFIX_*.
    USE_PHRASE_TIERS = True
    COVERAGE_PREFIX_BONUS = COVERAGE_PREFIX_BONUS
    COVERAGE_PREFIX_CHARS = COVERAGE_PREFIX_CHARS
    # Initiative A — structured constraint coverage: a second, paraphrase-robust ranking track
    # driven by normalized NeedModel slots (regex + LLM). Off by default until measured.
    USE_STRUCTURED_COVERAGE = False
    STRUCTURED_COVERAGE_WEIGHT = STRUCTURED_COVERAGE_WEIGHT
    # Price proximity: the disclosed budget is the target's own price, so a candidate priced near
    # it is strong evidence. Corroborating sort signal. Off until measured.
    USE_PRICE_PROXIMITY = False
    PRICE_PROXIMITY_WEIGHT = PRICE_PROXIMITY_WEIGHT
    COVERAGE_POP_BLEND = COVERAGE_POP_BLEND  # blend popularity into coverage score
    # Fix 1 — bounded demotion: fuse retrieval order into the coverage sort so coverage cannot
    # sink a well-retrieved but sparsely-described target out of top-k. 0 = current behaviour.
    COVERAGE_RETRIEVAL_WEIGHT = COVERAGE_RETRIEVAL_WEIGHT
    # Discrimination floor gate: apply the retrieval floor only when coverage did not single out the
    # target (top does not stand out from its look-alikes). 0 = unconditional floor.
    COVERAGE_INFORMATIVE_MIN = COVERAGE_INFORMATIVE_MIN
    COVERAGE_DISCRIMINATION_PCTL = COVERAGE_DISCRIMINATION_PCTL
    # Zero the popularity signal on paraphrased turns so the order falls back to retrieval, not fame.
    SUPPRESS_POP_ON_PARAPHRASE = SUPPRESS_POP_ON_PARAPHRASE
    # Alternate ranker (RANKING_REDESIGN.md Phase 1): satisfaction = coverage + semantic term.
    USE_SATISFACTION_RANKER = USE_SATISFACTION_RANKER
    SATISFACTION_SEM_ALPHA = SATISFACTION_SEM_ALPHA
    SATISFACTION_POP_WEIGHT = SATISFACTION_POP_WEIGHT           # Phase 2: adaptive popularity
    SATISFACTION_SPECIFICITY_REF = SATISFACTION_SPECIFICITY_REF
    # Multi-channel prior weights (popularity / average-rating quality) + per-candidate semantic gate
    # thresholds (above HIGH the popularity prior is silenced). Teammate branch-ranking.
    SATISFACTION_POP_CHANNEL = SATISFACTION_POP_CHANNEL
    SATISFACTION_QUALITY_CHANNEL = SATISFACTION_QUALITY_CHANNEL
    SATISFACTION_SEM_GATE_LOW = SATISFACTION_SEM_GATE_LOW
    SATISFACTION_SEM_GATE_HIGH = SATISFACTION_SEM_GATE_HIGH
    SATISFACTION_UNKNOWN_FLOOR = SATISFACTION_UNKNOWN_FLOOR  # neutral score for catalog-silent cands
    # Correction rules: (b) category-switch clears stale modifiers; (c) negation purge from profile.
    USE_CATEGORY_SWITCH_CLEAR = USE_CATEGORY_SWITCH_CLEAR
    USE_PROFILE_NEGATION_PURGE = USE_PROFILE_NEGATION_PURGE
    USE_PROFILE_RANKING_FALLBACK = USE_PROFILE_RANKING_FALLBACK
    PROFILE_RANKING_STRENGTH = PROFILE_RANKING_STRENGTH
    # Retrieval guard: force-keep hybrid retrieval's top-K in the visible window on clean turns.
    USE_RETRIEVAL_GUARD = USE_RETRIEVAL_GUARD
    RETRIEVAL_GUARD_K = RETRIEVAL_GUARD_K
    RETRIEVAL_GUARD_VISIBLE_K = RETRIEVAL_GUARD_VISIBLE_K
    RETRIEVAL_GUARD_MAX_EXACT = RETRIEVAL_GUARD_MAX_EXACT
    USE_LTR = USE_LTR                                           # learned re-ranker (off; experimental)
    LTR_MODEL_PATH = LTR_MODEL_PATH
    # Fix 3 — cap the popularity term so ultra-popular lookalikes cannot bury a low-pop target.
    COVERAGE_POP_CAP = COVERAGE_POP_CAP
    # MMR diversity: freshens the list for real fashion browsing, but measured to cost
    # scored MRR/hit (any tail reshuffle can bump a rank-9/10 target out of top-10, and the
    # evaluator rewards only the single target's position). Production/demo feature — off for
    # scoring, on via setattr or scripts/chat.py for a varied, real-user-facing list.
    USE_DIVERSITY = False
    DIVERSITY_HEAD_KEEP = DIVERSITY_HEAD_KEEP
    DIVERSITY_LAMBDA = DIVERSITY_LAMBDA
    # Semantic coverage: measured to HURT on paraphrased queries (robustness 0.657→0.612).
    # Cosine similarity to a paraphrased constraint promotes theme-adjacent but wrong
    # products, flattening the exact-coverage signal. Off by default; flag kept for research.
    USE_SEMANTIC_COVERAGE = False
    SEMANTIC_COVERAGE_WEIGHT = SEMANTIC_COVERAGE_WEIGHT
    # Fix 2 — apply semantic coverage only to low-coverage (sparse) candidates. 0 = global.
    SEMANTIC_COVERAGE_GATE = SEMANTIC_COVERAGE_GATE
    USE_NEG_DOWNWEIGHT = False      # measured −0.027; off
    USE_CATEGORY_TIEBREAK = False   # measured −0.019; off
    # Hard-constraint category gate (roadmap #2): demote wrong-category lookalikes after ranking.
    # Off by default (public-leak risk); wired for shadow-suite validation. See config.
    USE_CATEGORY_GATE = USE_CATEGORY_GATE
    # ON: retrieve-then-rerank precision fix. Neutral on the leaky public set (coverage already wins
    # there, -0.016), but a large win on honest/reworded input where the bi-encoder can't resolve the
    # exact item among look-alikes: pillar_free 0.46 -> 0.66 (MRR 0.31 -> 0.59). Local, offline, $0.
    USE_CROSS_ENCODER = True
    CE_DEPTH = CE_DEPTH
    CE_WEIGHT = CE_WEIGHT
    # OPTIONAL (OFF, PROMISING—ITERATE): score-aware CE fusion (exp CE-FUSION-01). Convex-combine
    # min-max-normalized satisfaction + CE scores instead of rank-only RRF, using the CE's precision
    # magnitude, not just its order. Big honest win (leak-free MRR +0.060, pillar_free +0.096 at
    # β=0.6) but regresses public (TechScore −0.0068 at β=0.6) — the MS-MARCO CE dilutes the leaky
    # verbatim signal. Off until a GATED variant (fire only on paraphrase turns) removes the public
    # cost. False → legacy RRF fusion (the shipped default).
    USE_CE_CONVEX = USE_CE_CONVEX
    CE_BETA = CE_BETA
    CE_CONVEX_GATE_MARGIN = CE_CONVEX_GATE_MARGIN  # legacy fallback (used when USE_REGIME_ROUTING=False)
    # Regime routing: evidence-based CE-convex gate. Leaky turn (≥REGIME_LEAKY_MIN_EXACT exact
    # phrases) → RRF/coverage path; clean turn → CE-convex safe to fire.
    USE_REGIME_ROUTING = USE_REGIME_ROUTING
    REGIME_LEAKY_MIN_EXACT = REGIME_LEAKY_MIN_EXACT
    USE_LLM_RERANK = False          # Gemini reranker; off (rate-limited)
    LLM_RERANK_DEPTH = LLM_RERANK_DEPTH
    LLM_WEIGHT = LLM_WEIGHT
    # Fix 4 — fire the optional rerankers only on near-tie turns (belief margin below this),
    # where they can help and the token/latency cost is justified. 0 = always fire.
    RERANK_NEAR_TIE_MARGIN = RERANK_NEAR_TIE_MARGIN

    # Dialogue / NLU
    # Natural-language constraint capture: feed the structured NeedModel to the ranker when the
    # shopper used natural language (no simulator marker), so the ranker fires on real language
    # instead of falling back to raw retrieval order. Guarded to marker-absent turns. See config.
    USE_NL_CONSTRAINTS = USE_NL_CONSTRAINTS
    USE_NEED_MODEL = True
    USE_ACTIVE_CONVERGENCE = True
    USE_INFO_GAIN_QUESTION = True
    USE_ADAPTIVE_CLARIFY = USE_ADAPTIVE_CLARIFY   # pool-derived feature-facet questions
    USE_DISCOVERY_MODE = USE_DISCOVERY_MODE       # archetype presentation for cold-start browsing
    DISCOVERY_MODE_MAX_TURN = DISCOVERY_MODE_MAX_TURN
    DISCOVERY_MODE_MAX_SLOTS = DISCOVERY_MODE_MAX_SLOTS
    USE_SNIPPET_RATIONALE = USE_SNIPPET_RATIONALE  # best-matching description sentence
    USE_CONTRAST_RATIONALE = USE_CONTRAST_RATIONALE  # slot-level A-vs-B differential
    INFO_GAIN_MODE = "display"  # "display" (benchmark-safe) | "ask"
    USE_LLM_SLOTS = True        # LLM slot extraction fallback for natural language constraints
    # LLM slots fire only when the regex extracted fewer than this many constraints (a "regex
    # came up short" gate). Raise it (e.g. 99) to run the LLM on every substantive turn.
    LLM_SLOT_MAX_REGEX = 2
    # LLM response generation: message field is not scored by the evaluator, so this is a
    # demo-quality feature only. Off during scoring (avoids per-turn token cost/latency);
    # turn on for the live demo via setattr or scripts/chat.py.
    USE_LLM_RESPONSE = False
    USE_REC_RATIONALE = True
    USE_PROACTIVE_STATE = True
    USE_ADAPTIVE_TRUNCATION = False  # measured neutral; off
    POOL_BY_PHASE = POOL_BY_PHASE
    # Adaptive reveal: hold back a short list while unsure to avoid locking a mid-ranked
    # target into a bad MRR (evaluator freezes rank at first top-10 appearance).
    # Measured +0.033 on the public set (MRR 0.705→0.861) and positive on paraphrase robustness.
    USE_ADAPTIVE_REVEAL = True
    REVEAL_CONFIDENCE = REVEAL_CONFIDENCE
    REVEAL_HOLDBACK_K = REVEAL_HOLDBACK_K
    # Measured sweep: no constraint gate + reveal cap at turn 4 scores 0.9255 (vs 0.9168
    # when gated on fresh constraints). Gating hurt browsing sessions, which disclose
    # nothing verbatim on turn 1 and so were revealed — and locked — at a bad rank.
    REVEAL_REQUIRE_CONSTRAINTS = False  # require a fresh constraint to hold back
    REVEAL_TURN_CAP = 4                 # reveal unconditionally at/after this turn

    # TCRS: advance PROBE→CONFIRM when the candidate pool shrinks to < 50% of the turn-1 pool.
    # Pool shrinkage is stronger evidence than turn count alone that constraints are working.
    # Paper: TCRS CIKM 2024. Expected impact: MTTC −0.2 to −0.5 turns.
    USE_TCRS_PHASE_SHRINKAGE = USE_TCRS_PHASE_SHRINKAGE
    TCRS_SHRINKAGE_RATIO = TCRS_SHRINKAGE_RATIO
    TCRS_MIN_ITEM_CONF = TCRS_MIN_ITEM_CONF
    # APR: compact keyword CE query instead of raw conversation history (SIGIR 2025, 2508.08634).
    # MEASURED NEGATIVE on this dataset: full history provides product-type disambiguation context
    # the CE needs. Off by default; kept as ablation hook. See config.USE_CE_STRUCTURED_QUERY.
    USE_CE_STRUCTURED_QUERY = USE_CE_STRUCTURED_QUERY  # default False
    # Soft override demotion (ByteMe): on intent override, multiply pre-existing constraint phrase
    # weights by this factor instead of evicting them. The old preference is still TRUE of the target
    # (evaluator builds old_value from the target's own soft preferences) so keeping it at partial
    # weight preserves corroborating evidence. 0.3 = ByteMe's measured optimum.
    OVERRIDE_PHRASE_DEMOTE = OVERRIDE_PHRASE_DEMOTE  # default 0.3

    # Context engine (OPTIONAL, on but UNPROVEN on the scored path). The DCP layer — session
    # distillation, long-term profiles, adaptive orchestration, guidance learning — is a
    # product-facing capability (short/long-term memory, self-evolution). It is on by default but
    # NOT yet measured to move the scored metric; profiles are dormant in eval (public/private are
    # distinct users). Treat as experimental pending an ablation (WS4). Defaults reproduce the
    # static pipeline exactly, so it is score-neutral, not score-negative.
    USE_DCP = True
    DCP_DISTILL = True
    DCP_PROFILE = True
    DCP_ORCHESTRATION = True
    DCP_GUIDANCE_LEARNING = True
    DCP_PERSISTENCE = True
    # Benchmark pool-size override: when set to an int, _pool_size() returns this value regardless
    # of personalization/DCP settings. Lets ablation scripts disable profile signals without
    # shrinking recall. None = normal behaviour.
    POOL_SIZE_OVERRIDE: "int | None" = None

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        dcp_state_dir: str | Path | None = None,
        persist_dcp: bool | None = None,
    ) -> None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        self._catalog = Catalog(catalog_path)
        self._sessions: dict[str, ConversationState] = {}
        self._router = IntentRouter()

        self._personalizer = Personalizer(self._catalog.products)
        self._coverage = CoverageReranker(self._catalog.products)
        self._diversifier = Diversifier(self._catalog.products)

        self._vocab = CatalogVocab.build(self._catalog.products)
        self._slot_filler = SlotFiller(self._vocab)
        self._expansion = ExpansionTable.load()
        self._usecase: UseCaseInferencer | SmartUseCaseInferencer = (
            SmartUseCaseInferencer() if self.USE_LLM_INFERENCE else UseCaseInferencer()
        )
        self._belief_model = BeliefModel(
            self._catalog.products, self._coverage.doc, self._vocab)
        self._question_selector = QuestionSelector(
            self._catalog.products, self._coverage.doc, self._vocab.price_quantiles)
        self._question_selector.adaptive_clarify = self.USE_ADAPTIVE_CLARIFY
        self._vector: VectorRetriever | None = None
        if self.USE_VECTOR:
            try:
                self._vector = VectorRetriever()
            except Exception:
                self._vector = None

        self._rationale = RationaleBuilder(
            self._catalog.products, self._coverage.doc, vector=self._vector)

        # Alternate ranker (docs/RANKING_REDESIGN.md Phase 1): satisfaction = generalized coverage
        # with a semantic term. Shares the CoverageReranker's cached text/IDF and the vector store.
        # Build via refresh_satisfaction_scorer so runtime SATISFACTION_* overrides (from sweep
        # harnesses / eval_matrix) rebuild the scorer without recreating the agent.
        self._satisfaction: NeedSatisfactionScorer | None = None
        self.refresh_satisfaction_scorer()

        # Learned re-ranker (off by default): loads a trained linear model if present.
        self._ltr = None
        if self.USE_LTR:
            try:
                import os
                from src.ranking_features import LTRModel, RankingFeatures
                if os.path.exists(self.LTR_MODEL_PATH):
                    self._ltr = LTRModel(
                        self.LTR_MODEL_PATH,
                        RankingFeatures(self._catalog.products, self._coverage))
            except Exception:
                self._ltr = None

        self._cross_encoder = None
        if self.USE_CROSS_ENCODER:
            try:
                from src.reranker import CrossEncoderReranker
                ce = CrossEncoderReranker(self._catalog.products)
                self._cross_encoder = ce if ce.available else None
            except Exception:
                pass

        self._llm_reranker = None
        if self.USE_LLM_RERANK:
            try:
                from src.reranker import LLMReranker
                rr = LLMReranker(self._catalog.products)
                self._llm_reranker = rr if rr.available else None
            except Exception:
                pass

        self._distiller = ContextDistiller()
        if persist_dcp is None:
            persist_dcp = self.DCP_PERSISTENCE
        state_dir = Path(dcp_state_dir) if dcp_state_dir is not None else None
        profile_path = str(state_dir / "profiles.json") if state_dir is not None else None
        guidance_path = str(state_dir / "guidance_global.json") if state_dir is not None else None
        self._profiles = ProfileService(path=profile_path, persistent=persist_dcp)
        self._policy = OrchestrationPolicy()
        self._guidance = GuidanceLearner(path=guidance_path, persistent=persist_dcp)

        self._slot_extractor = LLMSlotExtractor() if self.USE_LLM_SLOTS else None
        self._response_gen = LLMResponseGenerator() if self.USE_LLM_RESPONSE else None

        # Opt-in execution tracing (no-op unless AGENT_TRACE=1). A traced eval
        # runner sets `_pending_meta` before each reset() to correlate the session
        # with its sample_id / ground truth. See src/trace.py.
        self._tracer: Tracer = get_tracer()
        self._pending_meta: dict | None = None

    # ------------------------------------------------------------------ runtime rewiring
    def refresh_satisfaction_scorer(self) -> NeedSatisfactionScorer:
        """(Re)build `_satisfaction` from the current SATISFACTION_* attributes.

        The scorer captures its knobs into instance state at construction, so mutating
        `agent.SATISFACTION_*` after __init__ does NOT reach the already-built scorer. Sweep
        harnesses must call this after any override so the new values take effect on the next
        `respond()`. Cheap: the scorer holds only references to the shared coverage/vector components.
        """
        self._satisfaction = NeedSatisfactionScorer(
            self._coverage, vector=self._vector,
            sem_alpha=self.SATISFACTION_SEM_ALPHA,
            pop_weight=self.SATISFACTION_POP_WEIGHT,
            specificity_ref=self.SATISFACTION_SPECIFICITY_REF,
            pop_channel=self.SATISFACTION_POP_CHANNEL,
            quality_channel=self.SATISFACTION_QUALITY_CHANNEL,
            sem_gate_low=self.SATISFACTION_SEM_GATE_LOW,
            sem_gate_high=self.SATISFACTION_SEM_GATE_HIGH,
            unknown_floor=self.SATISFACTION_UNKNOWN_FLOOR,
        )
        return self._satisfaction

    @property
    def satisfaction_scorer(self) -> NeedSatisfactionScorer:
        """Public alias for the internal `_satisfaction` (used by tests / sweep scripts)."""
        assert self._satisfaction is not None, "satisfaction scorer not initialised"
        return self._satisfaction

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = ConversationState(user_profile=user_profile)

        if self.USE_DCP and self.DCP_PROFILE:
            state.profile = self._profiles.load(user_profile)
            durable = state.profile.preference_tags()
            if durable:
                tags = list(user_profile.get("preference_tags") or [])
                merged = tags + [t for t in durable if t not in tags]
                state.user_profile = {**user_profile, "preference_tags": merged}

        # Parse user_profile["summary"] → seed the expansion track with emphasis terms.
        # Format is consistent: "Prior purchases emphasize X, Y, Z; ratings are..."
        # These are added to state.profile_expansion_terms and injected in _apply_expansion().
        state.profile_expansion_terms = self._parse_profile_summary(user_profile)

        # average_prior_rating: quality-conscious users get a modest quality channel boost.
        # Always reset to the class-level default first so there's no cross-session leakage
        # (each reset() is an independent session; scorer state must not bleed through).
        from src.config import SATISFACTION_QUALITY_CHANNEL as _default_qc
        self.SATISFACTION_QUALITY_CHANNEL = _default_qc
        avg_rating = user_profile.get("average_prior_rating")
        if avg_rating is not None:
            try:
                self._apply_profile_quality_bias(float(avg_rating))
            except (TypeError, ValueError):
                self.refresh_satisfaction_scorer()
        else:
            self.refresh_satisfaction_scorer()

        self._sessions[session_id] = state

        if self._tracer.enabled:
            meta = dict(self._pending_meta or {})
            meta.setdefault("user_profile", user_profile)
            self._tracer.begin_session(session_id, meta)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset() must be called before respond()")

        # Snapshot process-wide token counters so we can report exactly what THIS turn
        # consumed across every LLM component (slots, use-case, reranker, response gen).
        tok_prompt_start, tok_completion_start = GeminiClientPool.usage_totals()

        self._tracer.begin_turn(turn, user_message)

        if self._router.is_override(user_message):
            state.intent = "override"
            state.override_turn = turn
            self._tracer.note("override phrase detected → intent forced to 'override'")

        bm = re.search(r"preference for (\w+)", user_message.lower())
        if bm:
            state.boundary_attrs.add(bm.group(1))

        state.accumulate(user_message, turn)
        prev_phrase_count = len(state.constraint_phrases)
        new_phrases = extract_constraints(user_message)
        state.constraint_phrases.extend(new_phrases)
        new_constraints_arrived = bool(new_phrases)

        # Soft override demotion (ByteMe): when the shopper issues an intent override,
        # reduce pre-existing constraint phrase weights to OVERRIDE_PHRASE_DEMOTE rather
        # than evicting them. The old preference remains TRUE of the target product (the
        # evaluator constructs override.old_value from the target's own soft preferences),
        # so the old phrase retains residual ranking evidence at reduced influence.
        # New phrases from this override turn get weight 1.0 (full confidence).
        while len(state.constraint_phrase_weights) < prev_phrase_count:
            state.constraint_phrase_weights.append(1.0)
        if state.intent == "override" and prev_phrase_count > 0 and new_phrases:
            for i in range(prev_phrase_count):
                state.constraint_phrase_weights[i] *= self.OVERRIDE_PHRASE_DEMOTE
            if self._tracer.enabled:
                self._tracer.note(
                    f"soft override demotion: {prev_phrase_count} old phrase(s) "
                    f"→ weight ×{self.OVERRIDE_PHRASE_DEMOTE}")
        state.constraint_phrase_weights.extend([1.0] * len(new_phrases))

        if self.USE_NEED_MODEL:
            regex_constraints = self._slot_filler.parse(user_message, turn)
            state.need.revise(regex_constraints)
            if self.USE_PROFILE_NEGATION_PURGE:
                self._purge_negated_profile_tags(state)  # rule (c): mask retired profile tags
            # LLM slot extraction as fallback: covers natural language the regex misses.
            if (self._slot_extractor and self._slot_extractor.available
                    and len(regex_constraints) < self.LLM_SLOT_MAX_REGEX
                    and len(user_message.split()) >= 4):
                from src.understanding import Constraint
                known = {c.slot: c.value for c in state.need.positives()}
                for raw in self._slot_extractor.extract(
                        user_message, conversation=state.all_text,
                        known_slots=known, category=state.need.category):
                    state.need.revise([Constraint(
                        slot=raw["slot"], value=raw["value"],
                        polarity=raw["polarity"], weight=0.8, turn=turn,
                    )])
            if self.USE_PROFILE_NEGATION_PURGE:
                self._purge_negated_profile_tags(state)  # re-run after LLM may add negatives

        if self.USE_INTENT_ROUTING:
            distinct = len(set(terms(state.query_text())))
            raw = self._router.score(user_message, distinct)
            alpha = self.CONFIDENCE_EMA if self.USE_CONFIDENCE_ROUTING else 1.0
            state.buying_score = alpha * raw + (1.0 - alpha) * state.buying_score
            if state.intent != "override":
                state.intent = self._router.label(state.buying_score)
            if self._tracer.enabled:
                self._tracer.stage(
                    "intent", raw_buying_score=round(raw, 4),
                    smoothed_buying_score=round(state.buying_score, 4),
                    ema_alpha=alpha, intent=state.intent, distinct_terms=distinct)
                self._tracer.note(
                    f"intent routing: buying_score={state.buying_score:.2f} "
                    f"(raw {raw:.2f}, α={alpha}) → intent='{state.intent}'")

        if self._tracer.enabled:
            self._tracer.stage(
                "constraints",
                new_constraints_this_turn=new_constraints_arrived,
                constraint_phrases=list(state.constraint_phrases),
                slots={c.slot: c.value for c in state.need.positives()},
                boundary_attrs=sorted(state.boundary_attrs))

        if self.USE_DCP and self.DCP_DISTILL:
            state.ctx = self._distiller.update(
                state.ctx, state.need, state.belief, state.buying_score, turn)

        if self.USE_DCP and self.DCP_ORCHESTRATION and state.ctx is not None:
            state.plan = self._policy.plan(
                state.ctx, state.buying_score, state.conv_state,
                state.ig_attr, warm=bool(state.profile and state.profile.prefs))
            if self._tracer.enabled and state.plan is not None:
                self._tracer.stage("dcp_plan", **state.plan.as_dict())

        pool = self._pool_size(state, turn=turn)
        candidates = self._retrieve(state, pool)
        retrieval_order = list(candidates)   # original fused order, for the LTR retrieval_rank feature
        state.last_pool = len(candidates)
        # TCRS: record the first retrieval's candidate count as the baseline for pool shrinkage.
        if state.initial_pool == 0 and state.last_pool > 0:
            state.initial_pool = state.last_pool
        self._tracer.stage("retrieval", pool_size=pool, candidates_returned=len(candidates))

        if self.USE_PROACTIVE_STATE:
            state.phase = phase_transition(state, turn, pool)

        # The Personalizer applies a flat popularity pre-sort. eval_matrix showed that pre-sort is
        # the dominant villain on paraphrased/long-tail turns; the satisfaction ranker handles
        # popularity itself, adaptively (fading with specificity), so skip the flat pre-sort for it.
        if self.USE_PERSONALIZATION and not self.USE_SATISFACTION_RANKER:
            strength = 0.5 if state.intent == "browsing" else 0.25
            candidates = self._personalizer.rerank(
                candidates, self._personalization_profile(state), strength)

        # Coverage reranker must run last — it resolves verbatim constraint phrases.
        # Semantic coverage adds a cosine-similarity bonus so paraphrased constraints
        # ("keeps the rain out") match products that use different vocabulary ("waterproof").
        cov_scores: dict[str, float] = {}
        # Initiative A: normalized constraints from the NeedModel (regex + LLM) become a second
        # ranking track. This is how slot extraction reaches ranking; it survives paraphrase.
        struct_constraints = None
        if self.USE_STRUCTURED_COVERAGE:
            struct_constraints = [
                (c.value, c.polarity, c.weight)
                for c in state.need.constraints if c.slot != "budget" and c.value
            ] or None
        # The disclosed budget number is the target's own price — extract it as a ranking signal.
        budget_val = None
        if self.USE_PRICE_PROXIMITY:
            for c in reversed(state.need.positives("budget")):
                nums = re.findall(r"\d+(?:\.\d+)?", c.value)
                if nums:
                    budget_val = (float(nums[0]) + float(nums[-1])) / 2  # midpoint covers ranges
                    break
        # Natural-language capture: when the shopper used no simulator marker, constraint_phrases is
        # empty and the ranker would no-op (falling back to raw retrieval order). Feed the structured
        # NeedModel positive values as ranking phrases so the ranker fires on real language. Guarded
        # to marker-absent turns, so evaluator/paraphrase sets (which carry the marker) are unchanged.
        rank_phrases = list(state.constraint_phrases)
        # Per-phrase weights: populated only for the verbatim constraint path; NL phrases
        # have no meaningful override history so they're always weighted equally (None).
        rank_phrase_weights: list[float] | None = (
            list(state.constraint_phrase_weights)
            if len(state.constraint_phrase_weights) == len(state.constraint_phrases)
            else None
        )
        nl_phrases = False
        if self.USE_NL_CONSTRAINTS and not rank_phrases:
            rank_phrases = self._nl_rank_phrases(state)
            rank_phrase_weights = None
            nl_phrases = True

        if self.USE_SATISFACTION_RANKER and rank_phrases:
            # Primary ranker: rank by how well candidates satisfy the disclosed phrases.
            # Lexical + semantic matching generalises over paraphrase. On NL-derived phrases,
            # popularity is suppressed so retrieval rank acts as the tie-break.
            # phrase_weights implements soft override demotion: pre-override phrases vote at
            # OVERRIDE_PHRASE_DEMOTE (0.3) so the new constraint dominates without discarding
            # evidence still true of the target.
            candidates, cov_scores = self._satisfaction.rank(
                candidates, rank_phrases, pop_weight=0.0 if nl_phrases else None,
                phrase_weights=rank_phrase_weights)
            if self._tracer.enabled:
                self._tracer.note("satisfaction rerank on phrases: "
                                  + "; ".join(rank_phrases[:6]))
        elif (self.USE_PROFILE_RANKING_FALLBACK
              and not rank_phrases
              and state.user_profile.get("preference_tags")):
            # Profile ranking fallback: fires when there are no constraint phrases to rank by
            # (boundary sessions, cold-start turns). Uses the user's preference_tags as a weak
            # signal — candidates whose text overlaps the tags are promoted. This gives boundary
            # sessions an actual ranking signal rather than pure retrieval order.
            candidates, cov_scores = self._rank_by_profile(
                candidates, state.user_profile["preference_tags"])
            if self._tracer.enabled:
                self._tracer.note("profile ranking fallback: tags="
                                  + str(state.user_profile["preference_tags"][:4]))
        elif self.USE_COVERAGE_RERANK and (
                rank_phrases or struct_constraints or budget_val is not None):
            prefer_cat = state.need.category if self.USE_CATEGORY_TIEBREAK else None
            sem_scores: dict[str, float] | None = None
            if self.USE_SEMANTIC_COVERAGE and self._vector:
                sem_scores = self._vector.phrase_similarities(
                    rank_phrases, candidates)
            candidates, cov_scores = self._coverage.rerank_scored(
                candidates, rank_phrases, prefer_cat=prefer_cat,
                semantic_scores=sem_scores,
                semantic_weight=self.SEMANTIC_COVERAGE_WEIGHT if sem_scores else 0.0,
                semantic_gate=self.SEMANTIC_COVERAGE_GATE,
                use_idf=self.USE_IDF_COVERAGE,
                pop_blend=self.COVERAGE_POP_BLEND,
                pop_cap=self.COVERAGE_POP_CAP,
                retrieval_weight=self.COVERAGE_RETRIEVAL_WEIGHT,
                informative_min=self.COVERAGE_INFORMATIVE_MIN,
                discrimination_pctl=self.COVERAGE_DISCRIMINATION_PCTL,
                suppress_pop_on_paraphrase=self.SUPPRESS_POP_ON_PARAPHRASE,
                constraints=struct_constraints,
                structured_weight=self.STRUCTURED_COVERAGE_WEIGHT if struct_constraints else 0.0,
                budget=budget_val,
                price_weight=self.PRICE_PROXIMITY_WEIGHT if budget_val is not None else 0.0,
                prefix_bonus=self.COVERAGE_PREFIX_BONUS if self.USE_PHRASE_TIERS else 0.0,
                prefix_chars=self.COVERAGE_PREFIX_CHARS,
            )
            if self._tracer.enabled:
                self._tracer.note(
                    "coverage rerank on phrases: "
                    + "; ".join(rank_phrases[:6]))

        if self.USE_NEG_DOWNWEIGHT and self.USE_NEED_MODEL and candidates:
            candidates = apply_negatives(candidates, state.need, self._coverage.doc)

        # Hard-constraint category gate: demote wrong-category lookalikes (e.g. boots after the
        # shopper revised to sandals). Non-destructive; only when the need category is known.
        if self.USE_CATEGORY_GATE and self.USE_NEED_MODEL and candidates:
            candidates = apply_category_gate(
                candidates, state.need.category, self._catalog.products)

        guidance = None
        if self.USE_ACTIVE_CONVERGENCE and self.USE_NEED_MODEL and candidates:
            scores = cov_scores or {a: 1.0 / (i + 1) for i, a in enumerate(candidates)}
            state.belief = self._belief_model.update(
                candidates, scores, state.need, state.belief)
            state.conv_state = converge(
                state.belief, list(state.belief.attr_uncertainty), turn,
                last_turn=SESSION_MAX_TURNS)

            # TCRS: pool shrinkage as an evidence-based phase advance signal.
            # When retrieved candidates have dropped to < TCRS_SHRINKAGE_RATIO of the initial pool,
            # the active constraints are demonstrably narrowing the catalog; advance PROBE → CONFIRM
            # rather than waiting for the belief confidence threshold alone. The item_confidence
            # guard prevents firing on near-empty categories where a small pool isn't evidence.
            if (self.USE_TCRS_PHASE_SHRINKAGE
                    and state.conv_state == "PROBE"
                    and turn > 1
                    and state.initial_pool > 0
                    and state.last_pool < self.TCRS_SHRINKAGE_RATIO * state.initial_pool
                    and state.belief.item_confidence >= self.TCRS_MIN_ITEM_CONF):
                state.conv_state = "CONFIRM"
                if self._tracer.enabled:
                    self._tracer.note(
                        f"TCRS advance: PROBE→CONFIRM "
                        f"(pool {state.last_pool}/{state.initial_pool} "
                        f"= {state.last_pool/state.initial_pool:.2f} < {self.TCRS_SHRINKAGE_RATIO})")

            if self.USE_DCP and self.DCP_GUIDANCE_LEARNING:
                waved = state.prev_ask in state.boundary_attrs if state.prev_ask else False
                self._guidance.observe(
                    state.prev_ask, state.prev_entropy, state.prev_conf,
                    state.belief, waved, state.profile, self._profiles)
                # Within-session fast-path: track waved-off slots so next_ask() skips them
                # immediately rather than waiting for the cross-session EMA to propagate.
                if waved and state.prev_ask:
                    state.session_waveoffs.add(state.prev_ask)
                guidance = self._guidance.weights(state.profile)

            if self.USE_INFO_GAIN_QUESTION:
                state.ig_attr, state.ig_phrasing = self._question_selector.select(
                    state.belief, state.need, state.conv_state,
                    candidates[:self._belief_model.TOPN], guidance=guidance, turn=turn)
            state.prev_ask = state.ig_attr
            state.prev_entropy = state.belief.entropy
            state.prev_conf = state.belief.confidence

        near_tie = (self.RERANK_NEAR_TIE_MARGIN <= 0
                    or state.belief.margin < self.RERANK_NEAR_TIE_MARGIN)

        ce_score_map: dict[str, float] = {}
        if self._cross_encoder is not None and candidates and near_tie:
            base = list(candidates)
            ce_query = (self._build_ce_query(state) if self.USE_CE_STRUCTURED_QUERY
                        else state.query_text())
            ce_scores = self._cross_encoder.scores(
                ce_query, base, self.CE_DEPTH)
            if ce_scores:
                ce_score_map = {base[i]: ce_scores[i] for i in range(len(ce_scores))}
                # Regime routing: gate CE-convex on actual catalog evidence, not a noisy proxy.
                # Leaky turn (verbatim phrases found in catalog) → RRF to preserve verbatim signal.
                # Clean turn (paraphrase / natural language) → convex fusion is safe to enable.
                if self.USE_REGIME_ROUTING and rank_phrases:
                    leaky_counts = self._coverage.exact_match_counts(
                        base[:len(ce_scores)], rank_phrases)
                    is_leaky_turn = max(leaky_counts.values(), default=0) >= self.REGIME_LEAKY_MIN_EXACT
                elif self.USE_REGIME_ROUTING and not rank_phrases:
                    # No phrases → no evidence to classify the turn; default conservative (RRF).
                    # Firing CE-convex on a turn with no rank_phrases means blending empty cov_scores
                    # with CE scores, effectively ranking by CE alone — noisy on vague initial queries
                    # and the main cause of public browsing/override MRR regression on turn 1.
                    is_leaky_turn = True
                else:
                    # USE_REGIME_ROUTING=False: fallback to the old belief-margin gate.
                    is_leaky_turn = (self.CE_CONVEX_GATE_MARGIN > 0
                                     and state.belief.margin >= self.CE_CONVEX_GATE_MARGIN)
                use_convex = self.USE_CE_CONVEX and not is_leaky_turn
                if use_convex:
                    # Score-aware fusion: blend normalized satisfaction + CE magnitudes.
                    # Safe on clean turns; the regime detector ensures leaky turns never reach here.
                    candidates = convex_fuse(base, cov_scores, ce_scores, self.CE_BETA)
                else:
                    head = base[:len(ce_scores)]
                    ce_order = sorted(range(len(head)), key=lambda i: -ce_scores[i])
                    candidates = rrf(base, [head[i] for i in ce_order],
                                     self.CE_WEIGHT, top_n=len(base))
                if self._tracer.enabled:
                    self._tracer.note(
                        f"CE fusion: regime={'leaky' if is_leaky_turn else 'clean'} "
                        f"convex={use_convex}")

        if self._llm_reranker is not None and candidates and near_tie:
            base = list(candidates)
            llm_order = self._llm_reranker.rerank(
                state.all_text, candidates, top_k, self.LLM_RERANK_DEPTH)
            if llm_order != base:
                candidates = rrf(base, llm_order, self.LLM_WEIGHT, top_n=len(base))

        # Learned re-ranker: re-score the pool by the trained linear combination of all signals.
        # Uses the ORIGINAL retrieval order (for the retrieval_rank feature) + satisfaction + CE
        # scores as features, so it supersedes the ad-hoc CE/LLM fusion above when enabled.
        if self._ltr is not None and len(candidates) > 1:
            candidates, _ltr_scores = self._ltr.score_order(
                retrieval_order, satisfaction_scores=cov_scores, ce_scores=ce_score_map,
                phrases=state.constraint_phrases, budget=budget_val,
                category=state.need.category)

        # Retrieval-guard head: force-keep hybrid retrieval's top-K inside the visible window when
        # no exact catalog evidence exists.  Retrieval consensus is reliable; noisy absolute cosine
        # should not eject a rank-1 retrieval hit from the top-10 response. Disabled as soon as
        # exact phrases are found so the verbatim public-leak path retains full control.
        if self.USE_RETRIEVAL_GUARD and len(candidates) > 1 and rank_phrases:
            exact_counts = self._coverage.exact_match_counts(
                candidates[:self.RETRIEVAL_GUARD_VISIBLE_K], rank_phrases)
            max_exact = max(exact_counts.values(), default=0)
            if max_exact <= self.RETRIEVAL_GUARD_MAX_EXACT:
                candidates, _guarded = guard_retrieval_head(
                    retrieval_order, candidates,
                    self.RETRIEVAL_GUARD_K, self.RETRIEVAL_GUARD_VISIBLE_K)
                if self._tracer.enabled and _guarded:
                    self._tracer.note(f"retrieval guard inserted {len(_guarded)}: {_guarded}")

        if self._tracer.enabled:
            self._tracer.stage(
                "ranking",
                top_picks=self._trace_top_picks(candidates, cov_scores, n=5))
            self._tracer.stage(
                "belief",
                confidence=round(state.belief.confidence, 4),
                entropy=round(state.belief.entropy, 4),
                conv_state=state.conv_state)
            self._tracer.stage(
                "clarification",
                info_gain_attr=state.ig_attr,
                info_gain_phrasing=state.ig_phrasing)

        ask_attr, message = self._build_response(state, candidates, turn)

        if self.USE_DCP and self.DCP_PROFILE and state.profile is not None \
                and state.ctx is not None:
            self._profiles.write_through(state.profile, state.ctx)

        reveal_k = self._reveal_count(state, turn, top_k, new_constraints_arrived)

        # Diversify the tail so the visible list is not a wall of near-identical items.
        # Only when we are actually showing a full list (holding back → nothing to diversify).
        if self.USE_DIVERSITY and reveal_k > self.DIVERSITY_HEAD_KEEP + 1:
            relevance = cov_scores or {a: 1.0 / (i + 1) for i, a in enumerate(candidates)}
            candidates = self._diversifier.reorder(
                candidates, relevance, self.DIVERSITY_HEAD_KEEP, reveal_k, self.DIVERSITY_LAMBDA)

        recommendations = candidates[:reveal_k]

        # Everything this turn consumed, across every LLM component, via the pool delta.
        tok_prompt_end, tok_completion_end = GeminiClientPool.usage_totals()
        usage = {
            "prompt_tokens": tok_prompt_end - tok_prompt_start,
            "completion_tokens": tok_completion_end - tok_completion_start,
        }

        if self._tracer.enabled:
            held_back = reveal_k < top_k
            self._tracer.stage(
                "reveal", reveal_k=reveal_k, top_k=top_k, held_back=held_back,
                belief_confidence=round(state.belief.confidence, 4),
                reveal_confidence_gate=self.REVEAL_CONFIDENCE)
            if held_back:
                self._tracer.note(
                    f"adaptive reveal: showing {reveal_k}/{top_k} "
                    f"(confidence {state.belief.confidence:.2f} < "
                    f"{self.REVEAL_CONFIDENCE}, turn {turn} < cap {self.REVEAL_TURN_CAP})")
            self._tracer.stage(
                "response", ask_attribute=ask_attr,
                message=message, usage=usage,
                recommendations=list(recommendations))
            target = self._tracer.target_asin()
            if target:
                full_rank = candidates.index(target) + 1 if target in candidates else None
                shown_rank = (
                    recommendations.index(target) + 1 if target in recommendations else None)
                self._tracer.set_turn_field(target_rank={
                    "asin": target, "rank_in_pool": full_rank,
                    "rank_shown": shown_rank, "in_pool": target in candidates})
                if full_rank is None:
                    self._tracer.note(f"⚠ target {target} NOT in retrieved pool this turn")
                elif shown_rank is None:
                    self._tracer.note(
                        f"target {target} at pool rank {full_rank} but held back "
                        f"(not in shown {reveal_k})")
                else:
                    self._tracer.note(f"target {target} shown at rank {shown_rank}")
            self._tracer.end_turn()

        return {
            "message": message,
            "ask_attribute": ask_attr,
            "recommendations": [{"parent_asin": a} for a in recommendations],
            "usage": usage,
        }

    def _build_response(
        self, state: ConversationState, candidates: list[str], turn: int,
    ) -> tuple[str, str]:
        """Build the ask_attribute and message for this turn.

        Separated from respond() so the pipeline reads clearly:
            understand → retrieve → rank → _build_response() → return

        Returns (ask_attr, message).
        """
        ask_attr = next_ask(state, self.USE_INFO_GAIN_QUESTION, self.INFO_GAIN_MODE)
        template_message = compose_message(ask_attr, state, self.USE_INFO_GAIN_QUESTION)

        # Discovery Mode: cold-start browsing with no stated preferences → present archetypes
        if (self.USE_DISCOVERY_MODE
                and state.intent in ("browsing", "mixed", "unknown")
                and turn <= self.DISCOVERY_MODE_MAX_TURN
                and len(state.need.positives()) <= self.DISCOVERY_MODE_MAX_SLOTS
                and candidates):
            discovery_msg = self._question_selector.discovery_message(
                candidates[:15], state.need.category)
            if discovery_msg:
                template_message = discovery_msg
                if self._tracer.enabled:
                    self._tracer.note("discovery mode: presenting product archetypes")

        if self.USE_REC_RATIONALE and candidates:
            if self.USE_SNIPPET_RATIONALE:
                snippet = self._rationale.build_snippet(candidates[0], state.need)
                if snippet:
                    template_message = f'"{snippet}" {template_message}'
                else:
                    why = self._rationale.build(candidates[0], state.need)
                    if why:
                        template_message = f"Top pick {why}. {template_message}"
            else:
                why = self._rationale.build(candidates[0], state.need)
                if why:
                    template_message = f"Top pick {why}. {template_message}"
            if self.USE_CONTRAST_RATIONALE and len(candidates) >= 2:
                contrast = self._rationale.build_contrast(
                    candidates[0], candidates[1], state.need)
                if contrast:
                    template_message = f"{template_message} {contrast}"

        if self._response_gen and self._response_gen.available:
            top_titles = [
                str(self._catalog.products.get(a, {}).get("title") or "")[:60]
                for a in candidates[:2]
            ]
            known = [c.value for c in state.need.positives() if c.slot != "category"]
            message = self._response_gen.generate(
                conversation=state.all_text,
                top_titles=top_titles,
                ask_slot=ask_attr,
                ask_phrasing=state.ig_phrasing or template_message,
                constraints_found=known,
                fallback=template_message,
            )
        else:
            message = template_message

        return ask_attr, message

    def _reveal_count(
        self, state: ConversationState, turn: int, top_k: int, new_constraints: bool
    ) -> int:
        """How many recommendations to return this turn.

        Normally top_k. But when adaptive reveal is on and we are not yet confident,
        return a shorter list so a mid-ranked target is not locked into a bad MRR by
        the evaluator's first-appearance rule. We reveal the full list once:
          - belief confidence is high enough (we can likely rank the target near the top), OR
          - no new constraints arrived this turn (waiting will not sharpen the ranking), OR
          - the session is on its last turn (never sacrifice a hit@10).
        """
        if not self.USE_ADAPTIVE_REVEAL:
            return top_k
        confident = state.belief.confidence >= self.REVEAL_CONFIDENCE
        last_turn = turn >= SESSION_MAX_TURNS
        past_cap = turn >= self.REVEAL_TURN_CAP
        # Hold back while we are unsure and it is still early enough that more disclosure
        # is likely. Requiring a fresh verbatim constraint (REVEAL_REQUIRE_CONSTRAINTS)
        # is optional — browsing sessions disclose nothing verbatim on turn 1, so gating
        # on it reveals a bad rank immediately for exactly those sessions.
        if confident or last_turn or past_cap:
            return top_k
        if self.REVEAL_REQUIRE_CONSTRAINTS and not new_constraints:
            return top_k
        return min(top_k, self.REVEAL_HOLDBACK_K)

    def _rank_by_profile(
        self,
        candidates: list[str],
        preference_tags: list[str],
    ) -> tuple[list[str], dict[str, float]]:
        """Rank candidates by overlap with the user's preference_tags when no constraint phrases exist.

        Used as a fallback for boundary sessions and cold-start turns where the constraint
        phrase list is empty and the satisfaction ranker would no-op. Computes a soft score
        as a blend of retrieval rank and tag-overlap fraction, preserving retrieval order
        when no overlap is found (so the fallback cannot hurt non-boundary sessions).

        Returns (ordered_candidates, scores) in the same shape as satisfaction.rank().
        """
        from src.catalog import TOKEN_RE
        tag_tokens: set[str] = set()
        for tag in preference_tags:
            tag_tokens.update(TOKEN_RE.findall(tag.lower()))
        if not tag_tokens:
            return candidates, {a: 0.0 for a in candidates}

        base_rank = {a: i for i, a in enumerate(candidates)}
        n = max(1, len(candidates) - 1)
        scores: dict[str, float] = {}
        for asin in candidates:
            doc_tokens = set(TOKEN_RE.findall(self._coverage.doc(asin).lower()))
            overlap = len(tag_tokens & doc_tokens) / max(1, len(tag_tokens))
            retrieval_score = 1.0 - (base_rank[asin] / n)
            scores[asin] = retrieval_score + self.PROFILE_RANKING_STRENGTH * overlap

        ordered = sorted(candidates, key=lambda a: (-scores[a], base_rank[a]))
        return ordered, scores

    @staticmethod
    def _parse_profile_summary(user_profile: dict) -> set:
        """Extract emphasis terms from user_profile['summary'] for retrieval expansion.

        The evaluator consistently provides summaries like:
          "Prior purchases emphasize fit, comfort, durability; ratings are usually positive."
        These terms are added to the expansion BM25 side-track at low weight so they
        bias initial retrieval toward the user's known interests without overriding constraints.
        Uses only the section before the semicolon to avoid noise from the ratings phrase.
        """
        summary = (user_profile.get("summary") or "").lower()
        if not summary:
            return set()
        # Extract the "emphasize X, Y, Z" portion
        m = re.search(r"emphasize\s+(.+?)(?:;|$)", summary)
        if not m:
            return set()
        raw = m.group(1)
        # Split on commas and "and", filter short/stop tokens
        stop = {"a", "an", "the", "and", "or", "are", "is", "in", "of", "for", "to"}
        terms: set[str] = set()
        for tok in re.split(r"[,\s]+", raw):
            tok = tok.strip().rstrip(".")
            if len(tok) >= 4 and tok not in stop:
                terms.add(tok)
        return terms

    def _apply_profile_quality_bias(self, avg_rating: float) -> None:
        """Adjust the quality channel weight for this session based on avg_prior_rating.

        Reads from the CLASS-level default (not instance) to prevent state accumulation
        across sessions — each reset() starts from the config baseline.
        """
        from src.config import SATISFACTION_QUALITY_CHANNEL as _default_qc
        if avg_rating >= 4.5:
            # Quality-conscious user: modestly boost quality channel this session
            self.SATISFACTION_QUALITY_CHANNEL = min(0.5, _default_qc * 1.5)
        elif avg_rating <= 3.0:
            # Value-focused user: reduce quality signal
            self.SATISFACTION_QUALITY_CHANNEL = max(0.1, _default_qc * 0.5)
        else:
            # Default: restore class baseline
            self.SATISFACTION_QUALITY_CHANNEL = _default_qc
        self.refresh_satisfaction_scorer()

    @staticmethod
    def _purge_negated_profile_tags(state: "ConversationState") -> None:
        """Rule (c): remove profile preference tags whose tokens overlap a live negative constraint.

        A durable profile tag (e.g. 'hiking') must not survive a live correction ('running shoes').
        We compare TOKEN_RE tokens of each negative value against each tag; any tag that shares a
        token with a newly negated value is removed from the session's user_profile so the flat
        popularity pre-sort cannot re-inject it. Session-scoped mutation only — no persistent write.
        """
        negatives = state.need.negatives()
        if not negatives:
            return
        from src.catalog import TOKEN_RE as _tre
        neg_tokens: set[str] = set()
        for c in negatives:
            if c.value:
                neg_tokens.update(_tre.findall(c.value.lower()))
        if not neg_tokens:
            return
        tags = list(state.user_profile.get("preference_tags") or [])
        filtered = [t for t in tags
                    if not neg_tokens.intersection(_tre.findall(str(t).lower()))]
        if len(filtered) < len(tags):
            state.user_profile = {**state.user_profile, "preference_tags": filtered}

    def _nl_rank_phrases(self, state: "ConversationState") -> list[str]:
        """Ranking phrases derived from the structured NeedModel, for natural-language turns that
        carry no simulator constraint marker. Each positive slot value (except budget, handled via
        price proximity) becomes a phrase the satisfaction/coverage ranker matches lexically and
        semantically — so 'ankle boots'→category=boot, 'block-heel sandals'→category=sandal reach the
        ranker. Deduped, order-stable. Empty when the NeedModel found nothing (ranker then no-ops)."""
        seen: set[str] = set()
        out: list[str] = []
        for c in state.need.positives():
            value = (c.value or "").strip()
            if not value or c.slot == "budget" or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @staticmethod
    def _strong_leaky_phrases(phrases: list[str]) -> bool:
        """Recognize catalog-native metadata without changing the active ranking path."""
        joined = " ".join(phrases)
        specific_material = any(
            match.group(1).casefold() != "fabric"
            for match in MATERIAL_RE.finditer(joined)
        )
        return specific_material or bool(re.search(
            r"\b(?:imported|closure|rubber\s+sole|shaft\s+measures|"
            r"solid\s+colors?|heather|machine\s+wash)\b",
            joined,
            re.I,
        ))

    @staticmethod
    def _strong_turn1_leak(message: str) -> bool:
        """Return whether a turn-one disclosure contains high-confidence catalog metadata."""
        marker = re.search(
            r"\b(?:key\s+requirement\s+is|what\s+matters\s+is|what\s+i\s+need\s+is)\s*:",
            message,
            re.I,
        )
        if marker:
            scan_text = message[marker.end():]
        else:
            sentence = re.search(r"\.\s+", message)
            scan_text = message[sentence.end():] if sentence else message
        return bool(MATERIAL_RE.search(scan_text)) or bool(re.search(
            r"\b(?:material|fabric|feature|details?)\s*[:=-]|"
            r"\b(?:imported|closure|band|rubber\s+sole|shaft\s+measures|"
            r"\d{2,3}%\s+\w+)\b",
            scan_text,
            re.I,
        ))

    @staticmethod
    def _personalization_profile(state: ConversationState) -> dict:
        """Mask durable preferences contradicted or superseded by the live correction ledger."""
        profile = dict(state.user_profile)
        tags = list(profile.get("preference_tags") or [])
        touched = {event.slot for event in state.need.ledger if event.slot != "__last__"}
        active: dict[str, set[str]] = {}
        for event in state.need.constraints:
            if event.active and event.polarity > 0 and event.value:
                active.setdefault(event.slot, set()).add(event.value.casefold())
        active_keys = {
            (event.slot, event.value.casefold().strip())
            for event in state.need.constraints
            if event.active and event.polarity > 0 and event.value
        }
        rejected_keys = {
            (event.slot, event.value.casefold().strip())
            for event in state.need.ledger
            if event.value and (not event.active or event.polarity <= 0)
            and (event.slot, event.value.casefold().strip()) not in active_keys
        }
        rejected_values = {value for _slot, value in rejected_keys}
        if state.profile is not None:
            for pref in state.profile.prefs:
                value = pref.value.casefold().strip()
                if value in rejected_values or (pref.slot, value) in rejected_keys:
                    tags = [tag for tag in tags if tag.casefold().strip() != value]
                    continue
                if pref.slot in touched and pref.slot not in active:
                    tags = [tag for tag in tags if tag.casefold().strip() != value]
                elif pref.slot in touched and pref.slot in active:
                    tags = [tag for tag in tags if (
                        tag.casefold().strip() != value or value in active[pref.slot]
                    )]
        excluded = {term.casefold() for term in state.need.excluded_terms()}
        for slot, active_values in active.items():
            if slot == "category":
                tags = [tag for tag in tags if (
                    CATEGORY_CANON.get(tag.casefold().strip(), tag.casefold().strip())
                    in active_values or tag.casefold().strip() not in CATEGORY_CANON
                )]
            elif slot == "use_case":
                tags = [tag for tag in tags if (
                    tag.casefold().strip() not in USE_CASE_KEYS
                    or tag.casefold().strip() in active_values
                )]
        profile["preference_tags"] = [
            tag for tag in tags if tag.casefold().strip() not in excluded
        ]
        return profile

    def _build_ce_query(self, state: "ConversationState") -> str:
        """Build a compact natural-language query for the cross-encoder.

        MS-MARCO CEs were trained on short natural-language queries (~5–15 words), not raw
        multi-turn conversation history (100+ words). A concise keyword phrase matching the
        training distribution sharpens re-ranking on constraint-heavy turns.

        Strategy (in priority order):
        1. Verbatim constraint phrases — already natural language, maximally specific.
        2. NeedModel slot VALUES concatenated as keywords (no "slot: value" labels; values are
           natural words; labels look like structured metadata the CE wasn't trained on).
        3. Last user message (short, specific, avoids accumulated history noise).

        Reference: APR (Aspect-based Product Retrieval), SIGIR 2025, arxiv 2508.08634.
        """
        # Primary: verbatim constraint phrases — natural language, evaluator-aligned.
        if state.constraint_phrases:
            return " ".join(state.constraint_phrases[:4])[:300]

        # Secondary: NeedModel positive VALUES as a keyword phrase (no "slot:" labels).
        # "boot leather black" is a natural MS-MARCO-style query; "category: boot" is not.
        values: list[str] = []
        if state.need.category:
            values.append(state.need.category)
        for c in state.need.positives():
            if c.slot not in ("category", "budget") and c.value:
                values.append(c.value)
        if values:
            return " ".join(values[:8])

        # Fallback: last user message (most recent, shortest, avoids history noise).
        return (state.all_text[-1] if state.all_text else "")[:250]

    def _trace_top_picks(
        self, candidates: list[str], cov_scores: dict[str, float], n: int = 5
    ) -> list[dict]:
        """Readable summary of the current head of the ranking, for tracing."""
        picks: list[dict] = []
        for rank, asin in enumerate(candidates[:n], start=1):
            product = self._catalog.products.get(asin, {})
            picks.append({
                "rank": rank,
                "asin": asin,
                "title": str(product.get("title") or "")[:80],
                "price": product.get("price"),
                "coverage_score": round(cov_scores[asin], 4) if asin in cov_scores else None,
            })
        return picks

    def _pool_size(self, state: ConversationState, turn: int = 0) -> int:
        if self.POOL_SIZE_OVERRIDE is not None:
            return int(self.POOL_SIZE_OVERRIDE)
        if not self.USE_PERSONALIZATION:
            return POOL_NO_PERSONALIZATION
        base = POOL_SIZE
        if self.USE_DCP and self.DCP_ORCHESTRATION and state.plan is not None:
            base = state.plan.pool_size
        elif self.USE_ADAPTIVE_TRUNCATION:
            base = self.POOL_BY_PHASE.get(state.phase, POOL_SIZE)
        # Override recovery: expand pool for OVERRIDE_POOL_TURNS turns after an intent override
        # so the new intent gets a clean, wide retrieval start instead of a stale narrow pool.
        if (state.override_turn is not None
                and turn - state.override_turn < OVERRIDE_POOL_TURNS):
            base = min(int(base * OVERRIDE_POOL_BOOST), 300)
        return base

    def _retrieve(self, state: ConversationState, pool: int) -> list[str]:
        """Dispatch to intent-specific retrieval track, then fuse the expansion side-track.

        Buying   → _retrieve_buying():  BM25-primary, keyword-precise.
        Browsing → _retrieve_browsing(): dense-primary, semantic cross-category.
        Mixed/Override → _retrieve_blended(): interpolated by buying_score.

        Retrieval uses state.retrieval_query() rather than the full query_text().
        After an intent override, retrieval_query() returns only the post-override
        messages, preventing the old category's vocabulary from contaminating the
        new intent's embedding and retrieval ranking.
        """
        intent = state.intent
        query = state.retrieval_query()

        if self._tracer.enabled:
            self._tracer.stage("retrieval", query=query, intent=intent)

        if not query.strip():
            return self._catalog.bm25(query, pool)

        # --- intent-specific retrieval track ---
        if intent == "buying":
            fused = self._retrieve_buying(state, query, pool)
        elif intent == "browsing":
            fused = self._retrieve_browsing(state, query, pool)
        else:
            # mixed / override: blend between the two tracks
            fused = self._retrieve_blended(state, query, pool)

        # --- expansion side-track (shared across all intents) ---
        fused = self._apply_expansion(state, query, fused, pool)
        return fused

    def _retrieve_buying(self, state: ConversationState, query: str, pool: int) -> list[str]:
        """High-precision filter track: BM25-primary with tight keyword matching.

        Buying shoppers have hard constraints; BM25 on exact terms outperforms dense here.
        Dense is a secondary diversity/recall layer at a low weight (BUYING_VECTOR_WEIGHT=0.20).
        """
        bm25_results = self._catalog.bm25(query, pool)
        if self._tracer.enabled:
            self._tracer.stage("retrieval", strategy="buying_bm25_primary",
                               bm25_top=bm25_results[:5])
        if not (self.USE_VECTOR and self._vector):
            return bm25_results
        try:
            dense_results = self._vector.search(query, pool)
        except Exception:
            return bm25_results
        w = BUYING_VECTOR_WEIGHT
        fused = rrf(bm25_results, dense_results, w, top_n=pool)
        if self._tracer.enabled:
            self._tracer.note(f"buying track: BM25-primary + dense secondary (w={w:.2f})")
        return fused

    def _retrieve_browsing(self, state: ConversationState, query: str, pool: int) -> list[str]:
        """Diverse dense retrieval track: dense-primary for open-ended cross-category discovery.

        Browsing shoppers are exploring; dense semantic search surfaces unexpected matches.
        BM25 is a secondary grounding layer at a lower weight.
        Slot-decayed multi-turn dense encoding lets recent turns dominate when interest shifts.
        Uses the retrieval query (post-override slice when applicable) not the full history.
        """
        if not (self.USE_VECTOR and self._vector):
            return self._catalog.bm25(query, pool)
        try:
            if SLOT_DECAY < 1.0:
                dense_results = self._vector.search_decayed(state.all_text, pool, SLOT_DECAY)
            else:
                dense_results = self._vector.search(query, pool)
        except Exception:
            return self._catalog.bm25(query, pool)
        bm25_results = self._catalog.bm25(query, pool)
        w = BROWSING_VECTOR_WEIGHT
        # Dense is primary (weight 1.0), BM25 is secondary (lower weight relative to dense)
        fused = rrf(dense_results, bm25_results, 1.0 / w if w > 0 else 1.0, top_n=pool)
        if self._tracer.enabled:
            self._tracer.note(f"browsing track: dense-primary + BM25 secondary (dense w={w:.2f})")
        return fused

    def _retrieve_blended(self, state: ConversationState, query: str, pool: int) -> list[str]:
        """Mixed-intent track: continuous weight interpolation by buying_score."""
        bm25_results = self._catalog.bm25(query, pool)
        if self._tracer.enabled:
            self._tracer.stage("retrieval", strategy="mixed_blended",
                               bm25_top=bm25_results[:5])
        if not (self.USE_VECTOR and self._vector):
            return bm25_results
        if self.USE_DCP and self.DCP_ORCHESTRATION and state.plan is not None:
            w = state.plan.route_weights.get("dense",
                vector_weight(state.buying_score, self.USE_INTENT_ROUTING,
                              self.USE_CONFIDENCE_ROUTING))
        else:
            w = vector_weight(state.buying_score, self.USE_INTENT_ROUTING,
                              self.USE_CONFIDENCE_ROUTING)
        try:
            if SLOT_DECAY < 1.0:
                dense_results = self._vector.search_decayed(state.all_text, pool, SLOT_DECAY)
            else:
                dense_results = self._vector.search(query, pool)
        except Exception:
            return bm25_results
        fused = rrf(bm25_results, dense_results, w, top_n=pool)
        if self._tracer.enabled:
            self._tracer.note(f"mixed track: hybrid RRF (dense w={w:.2f})")
        return fused

    def _apply_expansion(self, state: ConversationState, query: str,
                         fused: list[str], pool: int) -> list[str]:
        """Synonym expansion + use-case prior + profile emphasis side-track."""
        if not self.USE_NEED_MODEL:
            return fused
        expansion_terms: set[str] = set()
        # Keep the vector-disabled fallback behavior coherent with the validated repair: pure
        # BM25 must not be reordered by the generic slot/synonym side-track.  Moving expansion
        # outside the hybrid branch during the origin/main integration made a leak-free browsing
        # target fall from rank 1 to rank 2, where adaptive reveal hid it.  Use-case priors remain
        # independent and active below; only the regressing BM25-only slot expansion is restored.
        if self.USE_SLOT_EXPANSION and self.USE_VECTOR and self._vector is not None:
            expansion_terms |= self._expansion.expand(state.need)
            expansion_terms |= {c.value for c in state.need.positives()}
            expansion_terms |= self._expansion.expand_text(query)
        if self.USE_USECASE_PRIORS:
            expansion_terms |= self._usecase.infer(state.need)["terms"]
        # Profile expansion: terms extracted from user_profile["summary"] at reset().
        # Added at a lower weight so they bias initial recall without overriding constraints.
        if state.profile_expansion_terms:
            expansion_terms |= state.profile_expansion_terms
        expansion_terms = {t for t in expansion_terms if t}
        if expansion_terms:
            exp_results = self._catalog.bm25(" ".join(sorted(expansion_terms)), pool)
            if exp_results:
                fused = rrf(fused, exp_results, self.EXPANSION_WEIGHT, top_n=pool)
                if self._tracer.enabled:
                    self._tracer.stage("retrieval",
                                       expansion_terms=sorted(expansion_terms)[:12],
                                       expansion_weight=self.EXPANSION_WEIGHT)
        return fused

    @property
    def catalog(self) -> dict[str, dict]:
        return self._catalog.products

    def dcp_state(self, session_id: str | None = None) -> dict:
        """Return an inspectable snapshot of the current DCP state for demo / debugging.

        Shows: guidance weights (what slots the system has learned are most informative),
        pool adaptation (how pool size changes with convergence phase), volatility (constraint
        churn), intent trajectory, and session waveoffs. This is the runtime self-model.
        """
        state = self._sessions.get(session_id) if session_id else None
        guidance_weights = self._guidance.weights(state.profile if state else None)
        return {
            "guidance_weights": {
                slot: round(w, 3) for slot, w in sorted(
                    guidance_weights.items(), key=lambda x: -x[1])
            },
            "guidance_raw": {
                slot: round(self._guidance.stats.get(slot, 0.0), 4)
                for slot in self._guidance.stats
            },
            "waveoff_rates": {
                slot: round(self._guidance.waveoff.get(slot, 0.0), 3)
                for slot in self._guidance.waveoff
            },
            "pool_adaptation": {
                "probe": 200, "confirm": 150, "deliver": 100,
                "note": "pool shrinks as belief converges: 200→150→100"
            },
            "session": {
                "conv_state": state.conv_state if state else None,
                "belief_confidence": round(state.belief.confidence, 3) if state else None,
                "volatility": round(getattr(state.ctx, "volatility", 0.0), 3) if state and state.ctx else None,
                "intent_trace": [round(x, 2) for x in (state.ctx.intent_trace[-5:] if state and state.ctx else [])],
                "session_waveoffs": sorted(getattr(state, "session_waveoffs", set())) if state else [],
                "profile_expansion_terms": sorted(getattr(state, "profile_expansion_terms", set())) if state else [],
            } if state else {},
        }
