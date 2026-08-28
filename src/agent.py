"""Agent orchestrator — official challenge entry point (via starter/agent.py).

Wires together catalog, retrieval, ranking, NLU, dialogue, and context components.
No retrieval, ranking, or NLU logic lives here; see docs/ARCHITECTURE.md for component locations.

Feature flags on the Agent class are ablation toggles; the robustness harness overrides
them with setattr(Agent, k, v).
"""
from __future__ import annotations

import re
from pathlib import Path

from src.catalog import Catalog, terms
from src.config import (
    CE_DEPTH, CE_WEIGHT, CONFIDENCE_EMA, EXPANSION_WEIGHT,
    LLM_RERANK_DEPTH, LLM_WEIGHT, POOL_BY_PHASE, POOL_NO_PERSONALIZATION,
    POOL_SIZE, SLOT_DECAY,
)
from src.context_engine import (
    ContextDistiller, GuidanceLearner, OrchestrationPolicy, ProfileService,
)
from src.dialogue import (
    ConversationState, IntentRouter, compose_message, extract_constraints,
    next_ask, phase_transition,
)
from src.ranking import CoverageReranker, Personalizer
from src.retrieval import VectorRetriever, rrf, vector_weight
from src.understanding import (
    Belief, BeliefModel, CatalogVocab, ExpansionTable, NeedModel,
    QuestionSelector, RationaleBuilder, SlotFiller, UseCaseInferencer,
    apply_negatives, converge, missing_required,
)
from src.llm_inference import SmartUseCaseInferencer


class Agent:
    """Conversational shopping agent.

    Public API (fixed by the challenge spec):
        __init__(catalog_path: str | Path = "data/catalog.jsonl")
        reset(session_id: str, user_profile: dict) -> None
        respond(session_id: str, user_message: str, turn: int, top_k: int) -> dict

    Class attributes are ablation toggles; override with setattr(Agent, k, v).
    """

    # Retrieval
    USE_VECTOR = True           # dense BGE track (auto-off if cache absent)
    USE_SLOT_EXPANSION = True   # synonym expansion recall track
    EXPANSION_WEIGHT = EXPANSION_WEIGHT
    USE_USECASE_PRIORS = True   # occasion → implied attribute inference
    USE_LLM_INFERENCE = True    # LLM-powered use-case inference (falls back to static table)

    # Intent routing
    USE_INTENT_ROUTING = True
    USE_CONFIDENCE_ROUTING = True  # smooth buying_score interpolation
    CONFIDENCE_EMA = CONFIDENCE_EMA

    # Ranking
    USE_PERSONALIZATION = True
    USE_COVERAGE_RERANK = True      # verbatim constraint coverage (must run last)
    USE_NEG_DOWNWEIGHT = False      # measured −0.027; off
    USE_CATEGORY_TIEBREAK = False   # measured −0.019; off
    USE_CROSS_ENCODER = False       # measured neutral/negative; off
    CE_DEPTH = CE_DEPTH
    CE_WEIGHT = CE_WEIGHT
    USE_LLM_RERANK = False          # Gemini reranker; off (rate-limited)
    LLM_RERANK_DEPTH = LLM_RERANK_DEPTH
    LLM_WEIGHT = LLM_WEIGHT

    # Dialogue / NLU
    USE_NEED_MODEL = True
    USE_ACTIVE_CONVERGENCE = True
    USE_INFO_GAIN_QUESTION = True
    INFO_GAIN_MODE = "display"  # "display" (benchmark-safe) | "ask"
    USE_REC_RATIONALE = True
    USE_PROACTIVE_STATE = True
    USE_ADAPTIVE_TRUNCATION = False  # measured neutral; off
    POOL_BY_PHASE = POOL_BY_PHASE

    # Context engine
    USE_DCP = True
    DCP_DISTILL = True
    DCP_PROFILE = True
    DCP_ORCHESTRATION = True
    DCP_GUIDANCE_LEARNING = True

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
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
        self._rationale = RationaleBuilder(self._catalog.products, self._coverage.doc)

        self._vector: VectorRetriever | None = None
        if self.USE_VECTOR:
            try:
                self._vector = VectorRetriever()
            except Exception:
                self._vector = None

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
        self._profiles = ProfileService()
        self._policy = OrchestrationPolicy()
        self._guidance = GuidanceLearner()

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = ConversationState(user_profile=user_profile)

        if self.USE_DCP and self.DCP_PROFILE:
            state.profile = self._profiles.load(user_profile)
            durable = state.profile.preference_tags()
            if durable:
                tags = list(user_profile.get("preference_tags") or [])
                merged = tags + [t for t in durable if t not in tags]
                state.user_profile = {**user_profile, "preference_tags": merged}

        self._sessions[session_id] = state

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

        if self._router.is_override(user_message):
            state.intent = "override"

        bm = re.search(r"preference for (\w+)", user_message.lower())
        if bm:
            state.boundary_attrs.add(bm.group(1))

        state.accumulate(user_message)
        state.constraint_phrases.extend(extract_constraints(user_message))

        if self.USE_NEED_MODEL:
            state.need.revise(self._slot_filler.parse(user_message, turn))

        if self.USE_INTENT_ROUTING:
            distinct = len(set(terms(state.query_text())))
            raw = self._router.score(user_message, distinct)
            alpha = self.CONFIDENCE_EMA if self.USE_CONFIDENCE_ROUTING else 1.0
            state.buying_score = alpha * raw + (1.0 - alpha) * state.buying_score
            if state.intent != "override":
                state.intent = self._router.label(state.buying_score)

        if self.USE_DCP and self.DCP_DISTILL:
            state.ctx = self._distiller.update(
                state.ctx, state.need, state.belief, state.buying_score, turn)

        if self.USE_DCP and self.DCP_ORCHESTRATION and state.ctx is not None:
            state.plan = self._policy.plan(
                state.ctx, state.buying_score, state.conv_state,
                state.ig_attr, warm=bool(state.profile and state.profile.prefs))

        pool = self._pool_size(state)
        candidates = self._retrieve(state, pool)
        state.last_pool = len(candidates)

        if self.USE_PROACTIVE_STATE:
            state.phase = phase_transition(state, turn, pool)

        if self.USE_PERSONALIZATION:
            strength = 0.5 if state.intent == "browsing" else 0.25
            candidates = self._personalizer.rerank(
                candidates, state.user_profile, strength)

        # Coverage reranker must run last — it resolves verbatim constraint phrases.
        cov_scores: dict[str, float] = {}
        if self.USE_COVERAGE_RERANK and state.constraint_phrases:
            prefer_cat = state.need.category if self.USE_CATEGORY_TIEBREAK else None
            candidates, cov_scores = self._coverage.rerank_scored(
                candidates, state.constraint_phrases, prefer_cat=prefer_cat)

        if self.USE_NEG_DOWNWEIGHT and self.USE_NEED_MODEL and candidates:
            candidates = apply_negatives(candidates, state.need, self._coverage.doc)

        guidance = None
        if self.USE_ACTIVE_CONVERGENCE and self.USE_NEED_MODEL and candidates:
            scores = cov_scores or {a: 1.0 / (i + 1) for i, a in enumerate(candidates)}
            state.belief = self._belief_model.update(
                candidates, scores, state.need, state.belief)
            state.conv_state = converge(
                state.belief, list(state.belief.attr_uncertainty), turn)

            if self.USE_DCP and self.DCP_GUIDANCE_LEARNING:
                waved = state.prev_ask in state.boundary_attrs if state.prev_ask else False
                self._guidance.observe(
                    state.prev_ask, state.prev_entropy, state.prev_conf,
                    state.belief, waved, state.profile, self._profiles)
                guidance = self._guidance.weights(state.profile)

            if self.USE_INFO_GAIN_QUESTION:
                state.ig_attr, state.ig_phrasing = self._question_selector.select(
                    state.belief, state.need, state.conv_state,
                    candidates[:self._belief_model.TOPN], guidance=guidance)
            state.prev_ask = state.ig_attr
            state.prev_entropy = state.belief.entropy
            state.prev_conf = state.belief.confidence

        if self._cross_encoder is not None and candidates:
            base = list(candidates)
            ce_scores = self._cross_encoder.scores(
                state.query_text(), base, self.CE_DEPTH)
            if ce_scores:
                head = base[:len(ce_scores)]
                ce_order = sorted(range(len(head)), key=lambda i: -ce_scores[i])
                candidates = rrf(base, [head[i] for i in ce_order],
                                 self.CE_WEIGHT, top_n=len(base))

        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if self._llm_reranker is not None and candidates:
            base = list(candidates)
            p0 = self._llm_reranker.prompt_tokens
            c0 = self._llm_reranker.completion_tokens
            llm_order = self._llm_reranker.rerank(
                state.all_text, candidates, top_k, self.LLM_RERANK_DEPTH)
            usage = {
                "prompt_tokens": self._llm_reranker.prompt_tokens - p0,
                "completion_tokens": self._llm_reranker.completion_tokens - c0,
            }
            if llm_order != base:
                candidates = rrf(base, llm_order, self.LLM_WEIGHT, top_n=len(base))

        ask_attr = next_ask(state, self.USE_INFO_GAIN_QUESTION, self.INFO_GAIN_MODE)
        message = compose_message(ask_attr, state, self.USE_INFO_GAIN_QUESTION)

        if self.USE_REC_RATIONALE and candidates:
            why = self._rationale.build(candidates[0], state.need)
            if why:
                message = f"Top pick {why}. {message}"

        if self.USE_DCP and self.DCP_PROFILE and state.profile is not None \
                and state.ctx is not None:
            self._profiles.write_through(state.profile, state.ctx)

        return {
            "message": message,
            "ask_attribute": ask_attr,
            "recommendations": [{"parent_asin": a} for a in candidates[:top_k]],
            "usage": usage,
        }

    def _pool_size(self, state: ConversationState) -> int:
        if not self.USE_PERSONALIZATION:
            return POOL_NO_PERSONALIZATION
        if self.USE_DCP and self.DCP_ORCHESTRATION and state.plan is not None:
            return state.plan.pool_size
        if self.USE_ADAPTIVE_TRUNCATION:
            return self.POOL_BY_PHASE.get(state.phase, POOL_SIZE)
        return POOL_SIZE

    def _retrieve(self, state: ConversationState, pool: int) -> list[str]:
        """BM25 + optional dense + optional expansion side-track, fused via RRF."""
        query = state.query_text()
        bm25_results = self._catalog.bm25(query, pool)

        if not (self.USE_VECTOR and self._vector) or not query.strip():
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

        if self.USE_NEED_MODEL:
            expansion_terms: set[str] = set()
            if self.USE_SLOT_EXPANSION:
                expansion_terms |= self._expansion.expand(state.need)
                expansion_terms |= {c.value for c in state.need.positives()}
                # expand_text catches words the slot filler doesn't extract as named
                # slots (e.g. "merino", "vegan") so the data-driven table fires on them
                expansion_terms |= self._expansion.expand_text(query)
            if self.USE_USECASE_PRIORS:
                expansion_terms |= self._usecase.infer(state.need)["terms"]
            expansion_terms = {t for t in expansion_terms if t}
            if expansion_terms:
                exp_results = self._catalog.bm25(" ".join(sorted(expansion_terms)), pool)
                if exp_results:
                    fused = rrf(fused, exp_results, self.EXPANSION_WEIGHT, top_n=pool)

        return fused

    @property
    def catalog(self) -> dict[str, dict]:
        return self._catalog.products
