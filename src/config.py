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
COVERAGE_TIE_BREAK: str = "pop"  # "pop" (popularity) or "base" (incoming order)

# ---------------------------------------------------------------------------
# Optional rerankers (off by default — measured neutral/negative)
CE_DEPTH: int = 50    # candidates the cross-encoder rescores
CE_WEIGHT: float = 1.0
LLM_RERANK_DEPTH: int = 20
LLM_WEIGHT: float = 0.3
LLM_MODEL: str = "gemini-2.5-flash-lite"

# ---------------------------------------------------------------------------
# Dialogue / clarification
# "other" matches any undisclosed constraint (highest yield) and is repeatable;
# the rest fill in only if the shopper hasn't waved them off.
ASK_PRIORITY: list[str] = ["other", "feature", "material", "color", "style", "size", "use_case"]

# Thresholds for the proactive phase-transition state machine
EXPLORE_TERM_THRESHOLD: int = 6     # distinct query terms below → explore (over-general)
DELIVER_TURN_THRESHOLD: int = 7     # turn ≥ this → deliver (enough signal)

# ---------------------------------------------------------------------------
# DCP (context engine)
PROFILE_STORE: str = "cache/profiles.json"
GUIDANCE_STORE: str = "cache/guidance_global.json"
