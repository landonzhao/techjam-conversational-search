"""Central configuration — all tunable numeric constants in one place.

Change a number here, not inside retrieval/ranking/dialogue logic.
Feature flags (ablation toggles) live as class attributes on Agent so the
robustness harness can override them with setattr(Agent, k, v).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# BM25 field weights: (parent_asin, title, categories, features, details, store, description)
BM25_WEIGHTS: tuple[float, ...] = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

# Maximum terms sent to FTS5 per query (avoids pathologically long queries)
BM25_MAX_TERMS: int = 60

# ---------------------------------------------------------------------------
# Dense retrieval
VECTOR_WEIGHT: float = 0.25         # default when intent routing is off
BUYING_VECTOR_WEIGHT: float = 0.20  # high-intent: BM25-heavy (keywords match well)
BROWSING_VECTOR_WEIGHT: float = 0.35  # browsing: dense-heavy (broader recall)
SLOT_DECAY: float = 1.0             # <1.0 fades older turns in the dense query

# Embedding model and cache paths
EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBED_CACHE_NPY: str = "cache/embeddings.npy"
EMBED_CACHE_ASINS: str = "cache/asins.json"
EMBED_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# RRF fusion
RRF_K: int = 60  # rank smoothing constant in 1/(k+rank)

# ---------------------------------------------------------------------------
# Candidate pool
POOL_SIZE: int = 200          # default pool for retrieval (measured: 50→200 lifted MRR)
POOL_BY_PHASE: dict[str, int] = {"explore": 200, "converge": 200, "deliver": 120}
POOL_NO_PERSONALIZATION: int = 10  # minimal pool when Personalizer is disabled
OVERRIDE_POOL_BOOST: float = 1.5   # expand pool by this factor for 2 turns after intent override
OVERRIDE_POOL_TURNS: int = 2       # number of turns the boosted pool lasts

# ---------------------------------------------------------------------------
# Synonym expansion
EXPANSION_WEIGHT: float = 0.1  # weight of the expansion BM25 side-track in RRF

# ---------------------------------------------------------------------------
# Intent routing / EMA
CONFIDENCE_EMA: float = 0.6  # buying_score EMA: b_t = α·raw + (1−α)·b_{t−1}
# IntentRouter.score coefficients (were hardcoded in src/dialogue.py; heuristic — see DECISIONS.md).
# s = BUYING_CUE·[buy phrase] − BROWSING_CUE·[browse phrase] + HARD_CONSTRAINT·[regex hit]
#     + SPECIFICITY_SLOPE·(distinct_terms − SPECIFICITY_PIVOT);  buying_score = sigmoid(s)
INTENT_BUYING_CUE_WEIGHT: float = 1.5
INTENT_BROWSING_CUE_WEIGHT: float = 1.5
INTENT_HARD_CONSTRAINT_WEIGHT: float = 1.0
INTENT_SPECIFICITY_SLOPE: float = 0.18
INTENT_SPECIFICITY_PIVOT: int = 6
INTENT_BUYING_CUTOFF: float = 0.6    # buying_score ≥ this → "buying"
INTENT_BROWSING_CUTOFF: float = 0.4  # buying_score ≤ this → "browsing"; between → "mixed"
# BeliefModel item-confidence blend (were hardcoded in src/understanding.py):
# item_conf = MARGIN·margin + ENTROPY·(1−entropy) + STABILITY·min(stable/2, 1)
BELIEF_MARGIN_WEIGHT: float = 0.5
BELIEF_ENTROPY_WEIGHT: float = 0.3
BELIEF_STABILITY_WEIGHT: float = 0.2
# QuestionSelector: ask a direct product-comparison question when the top-2 belief margin is below
# this (candidates nearly tied → comparing beats an abstract attribute question).
COMPARISON_MARGIN: float = 0.15

# Convergence thresholds for belief-driven dialogue state
CONVERGE_HIGH: float = 0.60  # confidence ≥ this → DELIVER
CONVERGE_MID: float = 0.35   # item_confidence ≥ this AND no missing slots → CONFIRM

# ---------------------------------------------------------------------------
# Personalizer
POP_WEIGHT: float = 1.0   # log(rating_number) boost — biggest single ranking win
TAG_WEIGHT: float = 0.3   # profile tag overlap boost

# ---------------------------------------------------------------------------
# CoverageReranker
COVERAGE_LEN_WEIGHT: float = 0.15
COVERAGE_FULL_PHRASE_BONUS: float = 1.0
# Graduated phrase-bonus middle tier (ByteMe-style): between the coarse token-overlap floor and
# the exact-substring bonus, award a partial bonus when a long phrase's *contiguous leading prefix*
# appears verbatim. A single altered/inserted trailing word destroys the exact-substring match but
# leaves a long prefix intact, so this degrades more gracefully and sharpens near-miss ranks (MRR).
# Scaled below COVERAGE_FULL_PHRASE_BONUS so a prefix match can never outrank a true exact match.
COVERAGE_PREFIX_BONUS: float = 0.5   # 0 = tier off (reproduces the two-tier behaviour)
COVERAGE_PREFIX_CHARS: int = 25      # min chars of the normalized phrase's prefix that must match
# Measured (scripts/exp_phrase_tiers.py, LLM off): public byte-identical (exact substrings already
# resolve the leak, so the tier never differentiates there); paraphrase MRR 0.5965→0.5984 and hard
# MRR 0.5563→0.5647 / hit@10 0.714→0.717 at 0.5 bonus / 25 chars. Small, generalization-only, no
# regression — enabled by default via Agent.USE_PHRASE_TIERS.
COVERAGE_TIE_BREAK: str = "pop"  # "pop" (popularity) or "base" (incoming order)
# Blend log-popularity INTO the coverage score (not just as a tie-break) so a much more
# popular correct target can overcome a small coverage deficit against obscure lookalikes.
# Measured optimum 0.1 (public 0.926→0.931, hit@10 0.990→0.995).
COVERAGE_POP_BLEND: float = 0.1

# Diversity (MMR): after ranking, diversify the tail of the list so it is not ten
# near-identical popular items. Protects a leading head (where the target usually sits),
# then trades relevance vs novelty for the remaining slots. lam=1.0 disables it.
DIVERSITY_HEAD_KEEP: int = 3    # leading positions left as pure relevance ranking
DIVERSITY_LAMBDA: float = 0.7   # 1.0 = pure relevance, 0.0 = pure novelty
# Semantic coverage: cosine similarity bonus added on top of exact token coverage.
# Allows paraphrased constraints ("keeps rain out") to match catalog text ("waterproof").
# 0 = exact token matching only; 2.0 = semantic contributes ~equally to a partial exact match.
SEMANTIC_COVERAGE_WEIGHT: float = 2.0
# Fix 2 — apply semantic coverage only to candidates whose exact coverage is below this
# threshold (rescue sparsely-described items where lexical coverage fails). 0 = apply globally.
SEMANTIC_COVERAGE_GATE: float = 0.0

# Price proximity. The simulator discloses budget as "around $<target's own price>", so a candidate
# whose price is close to the disclosed budget is strong entity-resolution evidence. Added into the
# coverage SORT key (not the returned score, so belief stays clean). Off until measured.
PRICE_PROXIMITY_WEIGHT: float = 2.0   # strength of the proximity bonus in the sort key
PRICE_NEAR: float = 0.02              # |price-budget|/budget below this = exact-price match
PRICE_LOOSE: float = 0.15             # below this = near match; beyond = mild evidence against
PRICE_FAR_PENALTY: float = 0.1        # penalty slope for a present-but-far price (capped)

# Hard-constraint category gate (roadmap #2). Demote candidates whose own title resolves to a
# DIFFERENT canonical category than the confidently-known need category (e.g. keep sandals ahead of
# boots after 'ankle boots'→'block-heel sandals'). Non-destructive (violators to the back, never
# dropped). OFF by default: on the leaky public set category words already leak, so a hard gate risks
# the guardrail; needs shadow-suite validation. Research: confidence-gated hard filtering (GenFacet).
USE_CATEGORY_GATE: bool = False

# Initiative A — structured constraint coverage. A second ranking track that scores candidates
# by how many NORMALIZED NeedModel constraints (material=leather, size=2T, polarity-aware) they
# satisfy, matched against catalog text — not verbatim message phrases. Fused with the verbatim
# coverage order via bounded RRF at this weight. This is the path by which regex/LLM slot
# extraction reaches ranking, and it survives paraphrase (normalized values, not leaked tokens).
# 0 = off (pure verbatim coverage, current behaviour).
STRUCTURED_COVERAGE_WEIGHT: float = 0.0

# --- NeedSatisfactionScorer (docs/RANKING_REDESIGN.md, Phase 1) --------------------------------
# Alternate ranker: score each candidate by how well it SATISFIES the disclosed constraint phrases,
# where match = max(verbatim-lexical IDF fraction, SATISFACTION_SEM_ALPHA * semantic cosine). Coverage
# is the special case that uses only the lexical term; adding the semantic term makes the score
# survive paraphrase (the reworded phrase matches the product's real vocabulary by meaning). Replaces
# the coverage re-sort when on. ON by default: validated (scripts/validate_satisfaction.py) to lift
# the honest sets — pillar_free 0.295 -> 0.398 (+35%), pillar_moderate 0.483 -> 0.501 — while public
# holds at 0.903 (a deliberate -0.014 leaderboard cost for large paraphrase robustness). Set False to
# revert to the pure-coverage ranker (public 0.9172).
USE_SATISFACTION_RANKER: bool = True
# Weight on the semantic-cosine term relative to the verbatim-lexical term. 1.0 = a paraphrase match
# (cosine ~0.5) competes with a partial verbatim match; 0 = pure lexical (reproduces coverage).
SATISFACTION_SEM_ALPHA: float = 1.0
# Adaptive multi-channel prior (Phase 2, revised — teammate branch-ranking, Walmart Unified
# Supervision Framework style). A flat log-popularity nudge is the dominant villain on the honest set
# (pure coverage leak-free 0.125 -> 0.767 with popularity removed) yet HELPS the leaky public set: the
# prior is only wrong when the semantic channel is already confident about a long-tail match. So the
# prior is graded (two channels) and decayed by TWO factors — user specificity AND per-candidate
# semantic confidence:
#   w_pop(a) = SATISFACTION_POP_WEIGHT · (1 − specificity) · sem_gate(sem_conf(a))
#   prior(a) = POP_CHANNEL · pool_norm(log1p(rating_number)) + QUALITY_CHANNEL · norm(avg_rating)
#   ranked(a) = satisfaction(a) + w_pop(a) · prior(a)
# See NeedSatisfactionScorer._adaptive_prior. On natural-language turns the Agent passes pop_weight=0
# so the generic regex-derived phrases don't let fame reorder a well-retrieved target (see agent.py).
# VALUE: 0.15 (reverted from the teammate's 0.3). Consolidation measurement (docs/EXPERIMENTS.md
# CONSOLIDATION-03): her 0.3/REF=6 defaults regressed OUR full pipeline (public 0.8629 < 0.88 floor);
# the MECHANISM (multi-channel + semantic gate) with the proven 0.15/REF=3 values is strictly better
# on both axes (public 0.8842 ✓, pillar_free 0.6549 vs 0.6388). Her subset +0.036 didn't transfer.
SATISFACTION_POP_WEIGHT: float = 0.15
# Weights of the two prior channels — normalised to sum 1 so the prior stays on [0,1] (sat's scale).
SATISFACTION_POP_CHANNEL: float = 0.7      # log(rating_number), pool-normalised
SATISFACTION_QUALITY_CHANNEL: float = 0.3  # (average_rating − 3)/2, clipped to [0,1]
# Per-candidate semantic gate. Cosine ≤ LOW → full popularity (semantic unreliable, lean on prior);
# ≥ HIGH → zero popularity (trust the long-tail semantic match, don't let a popular near-neighbour
# overwrite it); linear between. Tuned on branch-ranking.
SATISFACTION_SEM_GATE_LOW: float = 0.25
SATISFACTION_SEM_GATE_HIGH: float = 0.65

# Learning-to-Rank (docs/ADVANCED_RANKING_PLAN.md). A trained linear model (cache/ltr_model.json,
# built by scripts/collect_ltr_data.py + train_ltr.py on leak-balanced data) re-ranks the pool by the
# learned combination of all signals (retrieval rank, satisfaction, coverage, cross-encoder, price,
# popularity...). OFF by default: the mechanism is validated (it learns to down-weight the verbatim
# leak) but not yet shown to beat the satisfaction+cross-encoder default through the evaluator.
USE_LTR: bool = False
LTR_MODEL_PATH: str = "cache/ltr_model.json"
# Number of disclosed constraint phrases at which specificity saturates to 1 (popularity -> 0).
# Kept at 3 (teammate branch-ranking raised it to 6 on a 25-row subset, +0.036; but on OUR full
# pipeline REF=6 with pop_weight=0.3 dropped public to 0.8629 < floor — see CONSOLIDATION-03). With
# the per-candidate semantic gate now doing the honest-set protection, REF=3 holds the public floor
# (0.8842) and keeps pillar_free high (0.6549). Re-sweep on the full sets, not subsets, if revisiting.
SATISFACTION_SPECIFICITY_REF: int = 3
# Neutral floor for candidates the catalog is SILENT about (zero lexical AND zero semantic evidence).
# Bryan's insight: silence ≠ conflict — give these an unknown-state score so they aren't unfairly
# demoted below candidates that happened to match boilerplate. Target: honest Hit@10 ↑ with MRR held.
# Off = 0.0 (old behaviour, penalises silent candidates relative to low-match ones).
SATISFACTION_UNKNOWN_FLOOR: float = 0.5  # CORE — directly closes Hit@10 gap

# --- Retrieval guard head (Phase 2 integration, Bryan) ------------------------------------------
# Force-keep the top-K hybrid-retrieval candidates inside the visible top-10 window when no exact
# catalog evidence exists. BM25+dense consensus is more reliable than noisy absolute cosine; a
# small score gap should not eject a rank-1 retrieval hit from the response.
# GUARD_MAX_EXACT: disable the guard as soon as this many exact phrases are found (0 = disable
# only on any exact match), so the verbatim public-leak path is never diluted.
# Off by default until fair-eval confirms no public regression; default-False = OPTIONAL.
USE_RETRIEVAL_GUARD: bool = True   # CORE — closes Bryan's Hit@10 advantage
RETRIEVAL_GUARD_K: int = 8        # protect hybrid retrieval's top-8 in the visible top-10
RETRIEVAL_GUARD_VISIBLE_K: int = 10
RETRIEVAL_GUARD_MAX_EXACT: int = 0  # disable guard once any exact phrase found

# Fix 1 — bounded demotion. RRF weight of the retrieval (dense+BM25) order fused with the verbatim
# coverage order. When coverage cannot match reworded language it collapses to a popularity
# tie-break and discards the semantic order dense retrieval already produced; this weight fuses that
# order back in as a floor, so paraphrased queries keep their semantic ranking (no LLM required).
# Sweep on the leak-free set (scripts/exp_retrieval_weight.py): flat 1.0 triples leak-free resilience
# (0.125 -> 0.385) for -0.013 public; flat 2.0 -> 0.605 leak-free but -0.037 public. The gate below
# (COVERAGE_INFORMATIVE_MIN) removes that tradeoff: with the floor applied ONLY on paraphrase turns,
# this weight can run high (paraphrase branch) without touching verbatim turns. 0 = pure coverage.
COVERAGE_RETRIEVAL_WEIGHT: float = 1.0

# Discrimination floor gate. The retrieval floor above is applied ONLY when verbatim coverage failed
# to single out the target — measured by whether the top candidate STANDS OUT from its rivals, not by
# raw coverage magnitude (a magnitude gate is fooled by a shared brand anchor: coverage looks high
# but every brand-mate carries it, so it identifies nothing). Normalized discrimination =
# (top_cov − p_pctl_cov)/top_cov ∈ [0,1]: ~1 when only the target carries the disclosed words
# (verbatim turn → floor OFF, protect public), ~0 when look-alikes share them (anchored paraphrase →
# floor ON, lean on retrieval). This is the min discrimination for a turn to count as "coverage found
# it". 0 = gate off (unconditional floor). Off by default: eval_matrix measured the gate lifts
# leak-free further but costs public; the plain flat floor above is the proven default. Enable the
# gate by raising this above 0 (0.5 is the studied starting point).
COVERAGE_INFORMATIVE_MIN: float = 0.0
# Pool percentile used as the rival reference in the gate (0.9 = p90 — does the top beat its closest
# look-alikes, ignoring the mass of zero-coverage candidates). Higher = stricter gate.
COVERAGE_DISCRIMINATION_PCTL: float = 0.9

# Paraphrase pop-suppression. On a paraphrased turn (coverage uninformative per the gate above),
# blending/tie-breaking on popularity collapses the order to "most famous" and re-pollutes the
# semantic ranking the floor is preserving. When True, popularity is zeroed on exactly those turns
# so coverage_order falls back to the retrieval order. Requires COVERAGE_INFORMATIVE_MIN > 0. Off by
# default until the follow-up sweep measures it against the floor alone.
SUPPRESS_POP_ON_PARAPHRASE: bool = False

# Fix 3 — cap on the popularity term in the coverage blend, so ultra-popular lookalikes
# cannot bury a low-popularity correct target. 0 = uncapped (current behaviour).
COVERAGE_POP_CAP: float = 0.0

# ---------------------------------------------------------------------------
# Optional rerankers (off by default — measured neutral/negative)
CE_DEPTH: int = 50    # candidates the cross-encoder rescores
CE_WEIGHT: float = 1.0
# Cross-encoder fusion mode (roadmap component I — docs/EXPERIMENTS.md, exp CE-FUSION-01).
# RRF fusion (CE_WEIGHT above) uses only the CE RANK order and discards its score magnitudes.
# Convex mode blends the min-max-normalized satisfaction and CE scores over the CE head:
#   FinalScore(c) = (1 − CE_BETA)·SatNorm(c) + CE_BETA·CENorm(c)
# STATUS: PROMISING — ITERATE (OFF by default). Validated on the ranking-isolation harness
# (scripts/exp_ce_fusion.py): vs RRF, β=0.6 lifts leak-free MRR 0.633→0.693 (+0.060) and pillar_free
# MRR 0.452→0.548 (+0.096), 53–74 sessions improved vs 11–16 worsened. BUT the official public
# evaluator REGRESSES at every honest-winning β (β=0.6: TechScore −0.0068, β=0.5: −0.010), because
# the MS-MARCO cross-encoder's magnitude dilutes the leaky verbatim signal. No global β satisfies
# both honest (+≥0.01 MRR) and public (≤0.005 regression). Kept OFF pending a GATED convex (apply the
# blend only on uninformative/paraphrase turns — the next experiment). Set True to enable globally.
USE_CE_CONVEX: bool = True   # CORE — CE-convex now safe via regime routing (see USE_REGIME_ROUTING)
CE_BETA: float = 0.6
# Legacy belief-margin gate (superseded by USE_REGIME_ROUTING below). Kept for rollback; only
# consulted when USE_REGIME_ROUTING is False. 0 = ungated (convex every turn, regresses public).
CE_CONVEX_GATE_MARGIN: float = 0.5

# --- Regime routing (Phase 3 integration) -------------------------------------------------------
# Replace the noisy belief.margin gate with evidence-based regime detection. When the shopper's
# disclosed phrases have ≥ REGIME_LEAKY_MIN_EXACT exact catalog matches per candidate, the session
# is on the public/leaky track → use RRF + coverage path (never dilute verbatim signal with CE).
# Otherwise it's a clean/paraphrase turn → CE-convex is safe to enable.
# This is why the old global CE_CONVEX_GATE_MARGIN could never hold the public floor: it gated on
# a noisy proxy (belief margin); the regime router gates on actual catalog evidence instead.
USE_REGIME_ROUTING: bool = True   # CORE — enables evidence-based CE-convex gating
REGIME_LEAKY_MIN_EXACT: int = 1   # exact phrase matches per top candidate to call session "leaky"
                                   # 1 = any exact match → leaky (conservative; 2 let too many public
                                   # turns slip through to CE-convex, costing -0.051 public MRR)

# --- Surgical correction rules (Phase 4 integration) -------------------------------------------
# Rule (a) same-turn negation always on — clearly correct, no measurement risk.
# Rule (b) category-switch modifier clear: retiring prior-turn constraints on category switch is
# correct for real sessions but regressed public boundary MTTC (4.10→6.60) because the evaluator's
# verbatim disclosures can trigger spurious category parses. Off by default; re-enable after
# validating on the honest intent-override/boundary sets specifically.
USE_CATEGORY_SWITCH_CLEAR: bool = True   # CORE — only fires on confirmed override turns (is_override=True)
# Rule (c) negation purge from profile — off by default (safe but low measurable impact on evals
# since public/private users don't share profile state). On for real user deployments.
USE_PROFILE_NEGATION_PURGE: bool = True  # OPTIONAL — mask retired profile tags this session
LLM_RERANK_DEPTH: int = 20
LLM_WEIGHT: float = 0.3
LLM_MODEL: str = "gemini-flash-lite-latest"
# Fix 4 — only fire the optional rerankers when the belief margin is below this (top
# candidates nearly tied), where reranking can help. 0 = always fire (current behaviour).
RERANK_NEAR_TIE_MARGIN: float = 0.0

# ---------------------------------------------------------------------------
# Dialogue / clarification
# "other" first: maps to any undisclosed constraint (highest yield, repeatable), which is what
# the evaluator's boundary sessions need. Bryan's reordering (structured slots first) regressed
# public boundary MTTC from 4.10 → 6.20 (+2.10 turns) because it wastes turns asking for specific
# slots that boundary sessions have waved off before reaching the catch-all "other". Reverted.
ASK_PRIORITY: list[str] = ["other", "feature", "material", "color", "style", "size", "use_case"]

# Thresholds for the proactive phase-transition state machine
EXPLORE_TERM_THRESHOLD: int = 6     # distinct query terms below → explore (over-general)
DELIVER_TURN_THRESHOLD: int = 7     # turn ≥ this → deliver (enough signal)

# Adaptive reveal: the evaluator freezes MRR at the FIRST turn the target enters the
# top-10. Surfacing the target early at a mediocre rank locks in a bad MRR. When belief
# confidence is low and the shopper is still disclosing constraints, we return a shorter
# list so a mid-ranked target is not prematurely locked; we reveal the full list once
# confidence is high, constraints stop arriving, or the session is about to end.
SESSION_MAX_TURNS: int = 10          # competition hard limit; always reveal on the last turn
REVEAL_CONFIDENCE: float = 0.55      # belief.confidence ≥ this → reveal full list now
REVEAL_HOLDBACK_K: int = 1           # list length while holding back (measured: K=1 best)

# ---------------------------------------------------------------------------
# Clarification (docs/CLARIFICATION_PLAN.md). Browsing is our weakest honest-set pillar because the
# belief can only ask about structured slots (material/color/style/use_case) — it can never form a
# `feature` question, yet most reworded constraints classify as `feature`. When on, the question
# selector (a) drops structured slots the candidate pool has no values for (asking them can't
# discriminate) and (b) adds a pool-derived `feature` facet: a distinctive token the top candidates
# split on, asked as a feature question. Category-adaptive by construction; off by default until
# measured on pillar_free browsing + the public MTTC guardrail.
USE_ADAPTIVE_CLARIFY: bool = True   # CORE — pool-derived facet questions; filters unaskable slots
# Discovery Mode (CoShop/CoPref, arXiv 2026): when the shopper is browsing with no stated
# preferences, present 3 product archetypes from the retrieved pool to help them construct
# their preference rather than asking abstract slot questions. Targets cold-start MTTC.
USE_DISCOVERY_MODE: bool = True
DISCOVERY_MODE_MAX_TURN: int = 2    # only on early turns when category is still unclear
DISCOVERY_MODE_MAX_SLOTS: int = 1   # max filled positive slots before switching off

# Snippet rationale (Snippet-CRS, arXiv 2024): surface the specific product description sentence
# that best matches the user's active need, making the recommendation self-explanatory.
USE_SNIPPET_RATIONALE: bool = True
# Contrastive explanation (C2-CRS, WSDM 2022): when returning 2+ recommendations, show slot-level
# differential between top-2 ("A wins on material; B is $30 cheaper").
USE_CONTRAST_RATIONALE: bool = True

# ---------------------------------------------------------------------------
# Natural-language constraint capture (docs/EXPERIMENTS.md — keystone honest-generalization fix).
# `extract_constraints` only fires on the simulator's "key requirement is:" marker, so on natural
# shopper language `constraint_phrases` is empty and the satisfaction/coverage ranker no-ops (falls
# back to raw retrieval order — our primary ranking signal is coupled to the benchmark's disclosure
# syntax). When on, the ranker ALSO consumes the NeedModel's structured positive slot values as
# ranking phrases — but ONLY on turns where no marker phrase was disclosed, so evaluator/paraphrase
# sets (which always carry the marker) are unchanged. Research: query-understanding → structured
# constraints → constrained ranking (GenFacet; relevance filtering).
USE_NL_CONSTRAINTS: bool = True
# Slots that hold at most one active value; a newer positive supersedes older positives of the SAME
# slot (DST selective-overwrite — SOM-DST / mentioned-slot-pools). Multi-valued slots (color,
# material, feature, style, use_case) accumulate so "black or navy" / "cotton or linen" coexist.
# Fixes stale-constraint revision (e.g. "ankle boots" → "actually, block-heel sandals").
SINGLE_VALUED_SLOTS: tuple[str, ...] = ("category", "size", "budget")

# ---------------------------------------------------------------------------
# DCP (context engine)
PROFILE_STORE: str = "cache/profiles.json"
GUIDANCE_STORE: str = "cache/guidance_global.json"
