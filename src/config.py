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
# Legacy fallback retained for callers that explicitly request the old policy. Benchmark code must
# use Agent.POOL_SIZE_OVERRIDE instead: disabling popularity/profile signals must not shrink recall.
POOL_NO_PERSONALIZATION: int = 10

# ---------------------------------------------------------------------------
# Synonym expansion
EXPANSION_WEIGHT: float = 0.1  # weight of the expansion BM25 side-track in RRF

# ---------------------------------------------------------------------------
# Intent routing / EMA
CONFIDENCE_EMA: float = 0.6  # buying_score EMA: b_t = α·raw + (1−α)·b_{t−1}

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
# the coverage re-sort when on. Off by default until it beats coverage on eval_matrix (pop-ablated).
USE_SATISFACTION_RANKER: bool = False
# Weight on the semantic-cosine term relative to the verbatim-lexical term. 1.0 = a paraphrase match
# (cosine ~0.5) competes with a partial verbatim match; 0 = pure lexical (reproduces coverage).
SATISFACTION_SEM_ALPHA: float = 1.0
# Adaptive popularity (Phase 2). eval_matrix showed popularity is the dominant villain on the honest
# set (pure coverage leak-free 0.125 -> 0.767 with popularity removed) but HELPS the leaky public set.
# So blend popularity as a prior weighted w_pop = SATISFACTION_POP_WEIGHT * (1 - specificity), where
# specificity rises with how much discriminating signal the shopper disclosed: fame breaks ties when
# the turn is vague, and fades to ~0 once the need is specific (so the long-tail target is not buried).
SATISFACTION_POP_WEIGHT: float = 0.3
# Number of disclosed constraint phrases at which specificity saturates to 1 (popularity -> 0).
SATISFACTION_SPECIFICITY_REF: int = 3

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
LLM_RERANK_DEPTH: int = 20
LLM_WEIGHT: float = 0.3
LLM_MODEL: str = "gemini-flash-lite-latest"
# Fix 4 — only fire the optional rerankers when the belief margin is below this (top
# candidates nearly tied), where reranking can help. 0 = always fire (current behaviour).
RERANK_NEAR_TIE_MARGIN: float = 0.0

# ---------------------------------------------------------------------------
# Dialogue / clarification
# "other" matches any undisclosed constraint (highest yield) and is repeatable;
# the rest fill in only if the shopper hasn't waved them off.
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
USE_ADAPTIVE_CLARIFY: bool = False

# ---------------------------------------------------------------------------
# DCP (context engine)
PROFILE_STORE: str = "cache/profiles.json"
GUIDANCE_STORE: str = "cache/guidance_global.json"
