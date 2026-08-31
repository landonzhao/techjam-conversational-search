# Decision, Heuristic & Formula Registry

> Central registry of **every hand-chosen number** in the system: weights, thresholds, gates, cutoffs,
> top-K values, decay rates, and scoring formulas. Companion to
> [architecture.md](../architecture.md) — that document explains the components; this one accounts for
> their tunables.
>
> **Origin** classifies where the value came from:
> `empirical` (found by a logged sweep/experiment) · `heuristic` (reasoned, not swept) ·
> `inherited` (baseline/library default) · `arbitrary` (chosen for plausibility, untested).
>
> **Status:** `validated` (an experiment shows it helps) · `provisional` (on, but weakly evidenced) ·
> `arbitrary` (untested) · `off` (not on the scored path).
>
> When you change any value here, update the `config.py` comment **and** this row in the same commit.
> When you add a new tunable, add a row here. Undocumented magic numbers are treated as bugs.

**Legend for "Metric":** HR = Hit Rate@10 · MRR · MTTC/Eff = Efficiency · — = none / indirect.

---

## 1. Retrieval

### 1.1 BM25 (`src/catalog.py`, `config.py`)

| Name | Location | Value | Formula / role | Purpose | Origin | Metric | Sensitivity | Status | Next experiment |
|---|---|---|---|---|---|---|---|---|---|
| `BM25_WEIGHTS` | config.py:11 | `(0,6,4,2.5,2.5,1.5,1)` | per-column FTS5 bm25 weights `(asin,title,cats,feat,details,store,desc)` | title/category matches count most | heuristic | HR | high — title-heavy favors keyword-stuffed titles | provisional | grid-sweep field weights on leak-free recall |
| `BM25_MAX_TERMS` | config.py:14 | 60 | cap on OR-terms per query | avoid pathologically long FTS queries | heuristic | HR | low | provisional | measure recall vs cap on long multi-turn queries |
| BM25 `k1`,`b` | SQLite FTS5 | 1.2, 0.75 | Okapi defaults (not overridden) | term saturation / length norm | inherited | HR | unknown (never tuned) | arbitrary | expose and sweep if recall plateaus |
| query combiner | catalog.py:109 | `OR` of terms | high-recall union | recall over precision (ranking does precision) | heuristic | HR | medium | validated (recall ~99%) | — |

### 1.2 Dense retrieval (`src/retrieval.py`, `config.py`)

| Name | Location | Value | Role | Purpose | Origin | Metric | Sensitivity | Status |
|---|---|---|---|---|---|---|---|---|
| `EMBED_MODEL` | config.py:24 | `BAAI/bge-small-en-v1.5` | 384-d sentence embedding | paraphrase-tolerant recall | inherited (strong small model) | HR/MRR | high (whole dense track) | validated |
| `EMBED_QUERY_PREFIX` | config.py:27 | `"Represent this sentence…"` | BGE-required query instruction | correct asymmetric encoding | inherited (model card) | HR | high if omitted | validated |
| similarity metric | retrieval.py:139 | cosine = dot (L2-norm) | `sim=q·p` | rank by meaning | inherited | HR | — | validated |
| `SLOT_DECAY` | config.py:21 | 1.0 (off) | recency weight `decay^(n-1-i)` in multi-turn dense query | let newest turn dominate on override | heuristic | HR | untested <1.0 | arbitrary |

### 1.3 Fusion (`src/retrieval.py`, `config.py`)

| Name | Location | Value | Formula | Purpose | Origin | Metric | Sensitivity | Status |
|---|---|---|---|---|---|---|---|---|
| `RRF_K` | config.py:31 | 60 | `Σ w/(k+rank+1)` | rank smoothing (standard RRF constant) | inherited (Cormack RRF paper uses 60) | HR | low near 60 | validated |
| `VECTOR_WEIGHT` | config.py:18 | 0.25 | dense weight when routing off | neutral midpoint | heuristic | HR | medium | provisional |
| `BUYING_VECTOR_WEIGHT` | config.py:19 | 0.20 | dense weight at buying_score=1 | buying → BM25-heavy | empirical | HR | medium | validated |
| `BROWSING_VECTOR_WEIGHT` | config.py:20 | 0.35 | dense weight at buying_score=0 | browsing → dense-heavy | empirical | HR | medium | validated |
| `EXPANSION_WEIGHT` | config.py:41 | 0.1 | RRF weight of synonym side-track | small recall widener, low blast radius | heuristic | HR | low | provisional |

**Formula — intent-aware dense weight** (`vector_weight`, confidence routing on):
```
w_dense(b) = BROWSING_VECTOR_WEIGHT + b·(BUYING_VECTOR_WEIGHT − BROWSING_VECTOR_WEIGHT)
           = 0.35 − 0.15·b,      b = buying_score ∈ [0,1]
```

### 1.4 Candidate pool

| Name | Location | Value | Purpose | Origin | Metric | Sensitivity | Status |
|---|---|---|---|---|---|---|---|
| `POOL_SIZE` | config.py:35 | 200 | pool depth into ranking | recall vs rank/CE cost | empirical ("50→200 lifted MRR") | HR/MRR | high below ~100 | validated |
| `POOL_BY_PHASE` | config.py:36 | explore/converge 200, deliver 120 | phase-adaptive depth | trim once converged | heuristic | MRR/Eff | low (off by default) | off |
| `POOL_NO_PERSONALIZATION` | config.py:37 | 10 | minimal pool when Personalizer off | ablation path | heuristic | — | off |

---

## 2. Intent routing (`src/dialogue.py`, `config.py`)

**Formula — buying score** (`IntentRouter.score`):
```
s = 1.5·[BUYING phrase present] − 1.5·[BROWSING phrase present]
    + 1.0·[hard-constraint regex hit] + 0.18·(distinct_terms − 6)
buying_score = σ(s) = 1/(1+e^-s)
label: ≥0.6 buying · ≤0.4 browsing · else mixed
EMA:   b_t = CONFIDENCE_EMA·raw + (1−CONFIDENCE_EMA)·b_{t−1}
```

| Name | Location | Value | Purpose | Origin | Metric | Sensitivity | Status |
|---|---|---|---|---|---|---|---|
| `INTENT_BUYING/BROWSING_CUE_WEIGHT` | config.py | ±1.5 | strong lexical evidence | arbitrary | MTTC/HR | medium | arbitrary (now in config) |
| `INTENT_HARD_CONSTRAINT_WEIGHT` | config.py | +1.0 | constraint ⇒ buying | arbitrary | MTTC | medium | arbitrary |
| `INTENT_SPECIFICITY_SLOPE/PIVOT` | config.py | 0.18 / 6 | more distinct terms ⇒ buying | arbitrary | MTTC | medium | arbitrary |
| `INTENT_BUYING/BROWSING_CUTOFF` | config.py | 0.6 / 0.4 | buying/browsing thresholds | heuristic | MTTC | medium | provisional |
| `CONFIDENCE_EMA` | config.py:45 | 0.6 | intent smoothing across turns | heuristic | MTTC | medium | provisional |

> ✅ Config hygiene (roadmap #5, exp FIXES-02): the four `score()` coefficients and the label cutoffs
> were moved from hardcoded literals in `dialogue.py` into `config.py` (values unchanged). Still
> untested/arbitrary in value, but now centralized and swept-able — no longer hidden.

---

## 3. Personalizer (`src/ranking.py`, `config.py`)

**Formula:** `sort key = incoming_rank − boost`, `boost = POP_WEIGHT·log1p(rating_number) +
strength·TAG_WEIGHT·|profile∩product terms|`, `strength = 0.25 buying / 0.5 browsing`.

| Name | Location | Value | Purpose | Origin | Metric | Sensitivity | Status |
|---|---|---|---|---|---|---|---|
| `POP_WEIGHT` | config.py:53 | 1.0 | log-popularity boost | empirical ("MRR 0.565→0.66") | HR/MRR | high (helps leaky, hurts honest) | validated but **corrosive on honest** |
| `TAG_WEIGHT` | config.py:54 | 0.3 | profile-tag overlap boost | heuristic | MRR | low | provisional |
| buying/browsing strength | ranking.py:391 | 0.25 / 0.5 | scale tag term by intent | arbitrary | MRR | low | arbitrary |

> Personalizer is **skipped entirely when `USE_SATISFACTION_RANKER` is on** (the default), because the
> flat popularity pre-sort buries long-tail targets on paraphrase. See §5.

---

## 4. CoverageReranker (`src/ranking.py`, `config.py`)

**Formula — verbatim coverage** (per candidate, over phrases `(toks, whole)`):
```
phrase_weight = 1 + COVERAGE_LEN_WEIGHT·|toks|
token_score   = (present/|toks|)·phrase_weight            # or IDF-weighted fraction if use_idf
+ COVERAGE_FULL_PHRASE_BONUS·phrase_weight   if `whole` ⊆ catalog_text
+ COVERAGE_PREFIX_BONUS·phrase_weight        elif whole[:COVERAGE_PREFIX_CHARS] ⊆ catalog_text
IDF(t) = log(N/(1+df_t)) + 1
default sort key = (−score − pop_blend·log1p(rating), base_rank)   # pop tie-break/blend
```

| Name | Location | Value | Role | Purpose | Origin | Metric | Sensitivity | Status |
|---|---|---|---|---|---|---|---|---|
| `COVERAGE_LEN_WEIGHT` | config.py:58 | 0.15 | longer phrase = more specific | reward specific matches | heuristic | MRR | low | provisional |
| `COVERAGE_FULL_PHRASE_BONUS` | config.py:59 | 1.0 | exact-substring bonus | exact phrase singles out target | heuristic | MRR | high on leaky | validated |
| `COVERAGE_PREFIX_BONUS` | config.py:65 | 0.5 | graduated middle tier | near-miss (one altered trailing word) | empirical (exp_phrase_tiers) | MRR | small | validated (generalization) |
| `COVERAGE_PREFIX_CHARS` | config.py:66 | 25 | min prefix chars to match | avoid trivial prefixes | heuristic | MRR | low | provisional |
| `COVERAGE_TIE_BREAK` | config.py:71 | "pop" | tie-break by popularity | break coverage ties | heuristic | MRR | medium | provisional |
| `COVERAGE_POP_BLEND` | config.py:75 | 0.1 | blend log-pop into score | popular target overcomes small deficit | empirical ("0.926→0.931") | MRR | medium | validated |
| `COVERAGE_POP_CAP` | config.py:170 | 0.0 (off) | cap pop term | stop ultra-popular lookalikes burying target | heuristic | MRR | — | off |
| `COVERAGE_RETRIEVAL_WEIGHT` | config.py:144 | 1.0 | retrieval floor RRF weight | keep sparse-but-retrieved target afloat | empirical (exp_retrieval_weight; leak-free 0.125→0.385 for −0.013 public) | HR/MRR | high on honest | validated |
| `COVERAGE_INFORMATIVE_MIN` | config.py:156 | 0.0 (off) | discrimination gate | apply floor only on paraphrase turns | empirical (eval_matrix) | HR/MRR | high | off (tradeoff) |
| `COVERAGE_DISCRIMINATION_PCTL` | config.py:159 | 0.9 | p90 rival reference in gate | measure "top stands out" not magnitude | heuristic | MRR | medium | off |
| `SUPPRESS_POP_ON_PARAPHRASE` | config.py:166 | False | zero pop on uninformative turns | stop collapse to "most famous" | heuristic | MRR | medium | off |
| `SEMANTIC_COVERAGE_WEIGHT` | config.py:85 | 2.0 | cosine bonus weight | paraphrase match | empirical (measured HURTS: 0.657→0.612) | MRR | high | **off (negative)** |
| `SEMANTIC_COVERAGE_GATE` | config.py:88 | 0.0 | apply sem only to low-coverage | rescue sparse items | heuristic | MRR | — | off |
| `STRUCTURED_COVERAGE_WEIGHT` | config.py:104 | 0.0 (off) | RRF weight of normalized-slot track | paraphrase-robust slot ranking | heuristic | MRR | — | off (unmeasured) |
| `USE_IDF_COVERAGE` | agent.py flag | False | IDF-weight coverage tokens | rare tokens discriminate | empirical (neutral) | MRR | low | off |

**Discrimination gate formula** (when `informative_min > 0`):
```
discrimination = (top_cov − p_pctl_cov) / top_cov       # ∈ [0,1]; ~1 verbatim, ~0 anchored paraphrase
uninformative  = top_cov ≤ 0  OR  discrimination < COVERAGE_INFORMATIVE_MIN
→ apply retrieval floor / suppress pop only when uninformative
```

**Price proximity** (`_price_prox`, config.py:93–96, off):
```
delta = |price − budget| / max(budget,1)
+1.0 if delta < PRICE_NEAR (0.02) · +0.4 if delta < PRICE_LOOSE (0.15) · else −PRICE_FAR_PENALTY(0.1)·min(delta,3)
```
`PRICE_PROXIMITY_WEIGHT = 2.0` (off).

---

## 5. NeedSatisfactionScorer — default ranker (`src/ranking.py`, `config.py`)

**Formula:**
```
match(phrase, cand) = max( lexical_IDF_fraction(phrase,cand),  SATISFACTION_SEM_ALPHA·max(0,cos) )
satisfaction(cand)  = Σ_phrase (1+0.15·|toks|)·match / Σ_phrase (1+0.15·|toks|)
specificity = min(1, |phrases| / SATISFACTION_SPECIFICITY_REF)
w_pop       = SATISFACTION_POP_WEIGHT·(1 − specificity)
ranked(cand)= satisfaction + w_pop·(pop(cand)/max_pop)
sort key    = (−ranked, base_rank)          # ties keep retrieval order
```

| Name | Location | Value | Purpose | Origin | Metric | Sensitivity | Status |
|---|---|---|---|---|---|---|---|
| `USE_SATISFACTION_RANKER` | config.py:115 | True | make satisfaction the default ranker | empirical (validate_satisfaction) | MRR/HR | high | **validated (default)** |
| `SATISFACTION_SEM_ALPHA` | config.py:118 | 1.0 | semantic-vs-lexical weight | empirical | MRR | high on honest | validated |
| `SATISFACTION_POP_WEIGHT` | config.py:124 | 0.15 | adaptive fame prior | empirical (sweet spot; holds public, lifts honest) | MRR/HR | medium | validated |
| `SATISFACTION_SPECIFICITY_REF` | config.py:134 | 3 | phrases at which pop→0 | heuristic | MRR | low | provisional |

**Evidence:** `scripts/validate_satisfaction.py` — leak-free pillar_free 0.295→0.398 (+35%),
pillar_moderate 0.483→0.501; public 0.9172→0.903 (deliberate −0.014). Coverage is the special case
`sem_alpha=0, pop_weight=0`.

---

## 6. Belief & convergence (`src/understanding.py`, `config.py`)

**Formula — belief** (over top `TOPN=20`, ranker scores):
```
margin  = (s0 − s_last)/s0
entropy = −Σ p_i ln p_i / ln(n),   p_i = s_i/Σs
item_confidence = 0.5·margin + 0.3·(1−entropy) + 0.2·min(stable_turns/2, 1)
attr_uncertainty[slot] = normalized entropy of that attribute over the head (≥0.5 if seen, 1.0 if none)
need_confidence = 1 − mean(attr_uncertainty)
confidence = min(item_confidence, need_confidence)
```

| Name | Location | Value | Purpose | Origin | Metric | Sensitivity | Status |
|---|---|---|---|---|---|---|---|
| `BeliefModel.TOPN` | understanding.py:481 | 20 | head size for belief | heuristic | MTTC | low | provisional |
| `BELIEF_MARGIN/ENTROPY/STABILITY_WEIGHT` | config.py | 0.5/0.3/0.2 | margin/entropy/stability mix | arbitrary | MTTC | medium | arbitrary (now in config, FIXES-02) |
| attr floor | understanding.py:526 | 0.5 / 1.0 | required-but-unknown stays askable | heuristic | MTTC | low | provisional |
| `CONVERGE_HIGH` | config.py:48 | 0.60 | confidence ⇒ DELIVER | heuristic | MTTC/Eff | high | provisional |
| `CONVERGE_MID` | config.py:49 | 0.35 | item-conf ⇒ CONFIRM | heuristic | MTTC | medium | provisional |
| `DECISION_WEIGHT` | understanding.py:545 | budget1.3/size1.2/material1.1/use_case1.0/category1.0/style0.9/color0.8 | slot info-gain priority | heuristic (pool-narrowing intuition) | MTTC | medium | provisional |
| `COMPARISON_MARGIN` | config.py | 0.15 | ask comparison when top-2 tied | arbitrary | MTTC | low | arbitrary (now in config, FIXES-02) |

**converge():** `DELIVER if confidence≥0.60 or turn≥10` · `CONFIRM if item_confidence≥0.35 and no
missing required` · `else PROBE`.

---

## 7. Clarification & ask policy (`src/config.py`, `src/dialogue.py`, `src/understanding.py`)

| Name | Location | Value | Purpose | Origin | Metric | Status |
|---|---|---|---|---|---|---|
| `ASK_PRIORITY` | config.py:187 | `[other,feature,material,color,style,size,use_case]` | fallback ask order ("other" = max yield) | heuristic | MTTC | provisional |
| `INFO_GAIN_MODE` | agent.py flag | "display" | info-gain phrased in message, ask_attribute="other" | heuristic (benchmark-safe) | MTTC | **provisional (benchmark-shaped)** |
| `USE_ADAPTIVE_CLARIFY` | config.py:210 | False | drop unanswerable slots + pool-derived feature facet | heuristic | MTTC | off (unmeasured) |
| feature-facet split | understanding.py:624–628 | present in ~half (strength≥0.5) | most discriminating pool token | heuristic | MTTC | off |
| `EXPLORE_TERM_THRESHOLD` | config.py:191 | 6 | <6 distinct terms ⇒ explore phase | heuristic | MTTC | provisional |
| `DELIVER_TURN_THRESHOLD` | config.py:192 | 7 | turn≥7 ⇒ deliver phase | heuristic | MTTC | provisional |

---

## 8. Adaptive reveal (`src/agent.py`, `config.py`)

**Rule:** reveal full `top_k` if `confidence ≥ REVEAL_CONFIDENCE` OR `turn ≥ SESSION_MAX_TURNS` OR
`turn ≥ REVEAL_TURN_CAP` OR (`REVEAL_REQUIRE_CONSTRAINTS` and no new constraint); else return
`min(top_k, REVEAL_HOLDBACK_K)`.

| Name | Location | Value | Purpose | Origin | Metric | Sensitivity | Status |
|---|---|---|---|---|---|---|---|
| `USE_ADAPTIVE_REVEAL` | agent.py flag | True | hold back while unsure | empirical (+0.033 public, MRR 0.705→0.861) | MRR | high | **validated** |
| `REVEAL_CONFIDENCE` | config.py:199 | 0.55 | confidence to reveal now | empirical | MRR | high | validated |
| `REVEAL_HOLDBACK_K` | config.py:200 | 1 | list length while holding back | empirical ("K=1 best") | MRR | high | validated |
| `REVEAL_TURN_CAP` | agent.py flag | 4 | reveal unconditionally by turn 4 | empirical (0.9255 vs 0.9168 gated) | MRR/Eff | high | validated |
| `REVEAL_REQUIRE_CONSTRAINTS` | agent.py flag | False | require fresh constraint to hold back | empirical (gating hurt browsing) | MRR | medium | validated (off) |
| `SESSION_MAX_TURNS` | config.py:198 | 10 | competition hard limit; always reveal last turn | inherited (rules) | HR | — | fixed |

---

## 9. Optional rerankers (`src/reranker.py`, `src/ranking_features.py`, `config.py`)

| Name | Location | Value | Purpose | Origin | Metric | Status |
|---|---|---|---|---|---|---|
| `USE_CROSS_ENCODER` | agent.py flag | True | precision rerank top 50 | empirical (honest 0.46→0.66, MRR 0.31→0.59; leaky −0.016) | MRR | validated (on) |
| `CE_DEPTH` | config.py:174 | 50 | candidates CE rescores | heuristic | MRR | provisional |
| `CE_WEIGHT` | config.py:175 | 1.0 | RRF weight of CE order | heuristic | MRR | provisional |
| `RERANK_NEAR_TIE_MARGIN` | config.py:181 | 0.0 (always) | fire rerankers only on near-ties | heuristic | MRR/Eff | provisional |
| `USE_CE_CONVEX` | config.py | False | score-aware convex CE fusion vs rank-only RRF | empirical (exp CE-FUSION-01) | MRR | **off — PROMISING—ITERATE** |
| `CE_BETA` | config.py | 0.6 | weight on CE vs satisfaction in convex fusion | empirical (isolation sweep) | MRR | validated-honest, off pending gate |

**Convex CE fusion formula** (`convex_fuse`, when `USE_CE_CONVEX`): over the CE head,
`FinalScore(c) = (1−CE_BETA)·minmax(satisfaction)[c] + CE_BETA·minmax(ce)[c]`; tail kept in
satisfaction order; deterministic tie-break on head index. **Status:** exp CE-FUSION-01 — β=0.6 lifts
leak-free MRR +0.060 / pillar_free +0.096 but regresses public TechScore −0.0068; no global β wins
both. Off by default; next step is a paraphrase-gated variant. See [EXPERIMENTS.md](EXPERIMENTS.md).
| `USE_LLM_RERANK` | agent.py flag | False | Gemini listwise rerank | rate-limited | MRR | off |
| `LLM_RERANK_DEPTH` | config.py:176 | 20 | candidates the LLM reorders | heuristic | MRR | off |
| `LLM_WEIGHT` | config.py:177 | 0.3 | RRF weight of LLM order | heuristic | MRR | off |
| `USE_LTR` | config.py:131 | False | learned linear rerank | experimental | MRR | off |

**LTR features** (`ranking_features.py`): retrieval_rank, satisfaction, coverage(IDF), cross_encoder,
log_popularity, avg_rating, price_proximity, category_match, specificity. Score =
`intercept + Σ wᵢ·((xᵢ−meanᵢ)/stdᵢ)` (standardized linear).

---

## 10. DCP / context engine (`src/context_engine.py`, `config.py`)

| Name | Location | Value | Purpose | Origin | Metric | Status |
|---|---|---|---|---|---|---|
| `USE_DCP` + family | agent.py flags | True | short/long memory, orchestration, guidance | — | none proven | **on but score-neutral / unproven** |
| `ContextDistiller.DECAY` | context_engine.py:65 | 0.9 | per-turn soft-constraint decay | heuristic | — | provisional |
| `MAX_CONSTRAINTS`/`MIN_KEEP`/`PRUNE_FLOOR` | context_engine.py:66–68 | 12 / 4 / 0.15 | salience pruning bounds | heuristic | — | provisional |
| `ProfileService.EMA` | context_engine.py:167 | 0.6 | write-through blend | heuristic | — | dormant in eval |
| `HALFLIFE_DAYS` | context_engine.py:168 | 45 | read-time time decay | heuristic | — | dormant |
| `GuidanceLearner.LAMBDA`/`EMA` | context_engine.py:299–300 | 0.5 / 0.3 | guidance strength / update rate | heuristic | MTTC | provisional |

> Every DCP default reproduces the static pipeline, so all are score-neutral. They earn Pillar III
> credit only if the WS4 ablation shows movement.

---

## 11. Evaluator constants (read-only contract — do not change)

| Name | Location | Value | Role |
|---|---|---|---|
| `MAX_TURNS` | local_evaluator.py:15 | 10 | session length limit |
| `TOP_K` | local_evaluator.py:16 | 10 | recommendations counted |
| TechnicalScore weights | local_evaluator.py:280 | 0.50 / 0.30 / 0.20 | HR / MRR / Efficiency blend |
| Efficiency | local_evaluator.py:279 | `clip((11−MTTC)/10,0,1)` | turn-efficiency term |
| MTTC miss penalty | local_evaluator.py:194 | `MAX_TURNS+1 = 11` | non-hit sessions |

---

## 12. Arbitrary values needing experiments (priority order)

These have **no supporting measurement** and are the first candidates for a sweep:

1. **IntentRouter coefficients** (±1.5, +1.0, 0.18/pivot-6, cutoffs 0.6/0.4) — hardcoded in
   `dialogue.py`. Move to `config.py`, sweep on MTTC + routing accuracy (intent test set).
2. **BeliefModel item-confidence blend** (0.5/0.3/0.2) — hardcoded in `understanding.py`. Calibrate
   against realized hit probability.
3. **Comparison-question margin** (0.15) and **DECISION_WEIGHT** map — sweep on MTTC.
4. `SATISFACTION_SPECIFICITY_REF` (3), `CE_DEPTH` (50), `COVERAGE_PREFIX_CHARS` (25),
   `RRF_K`-neighborhood — low-risk sensitivity checks.
5. Personalizer buying/browsing `strength` (0.25/0.5) — sweep against the pop-ablated column.

Use `scripts/eval_matrix.py` (relevance-vs-fame) and `evaluator/robustness.py` (paraphrase) for any
ranking-adjacent sweep; never accept a public-only improvement without checking the honest column.

---

## 13. Audit fixes (exp FIXES-02)

| Name | Location | Value | Purpose | Origin | Metric | Status |
|---|---|---|---|---|---|---|
| `USE_NL_CONSTRAINTS` | config.py | True | ranker consumes NeedModel on marker-absent (natural-language) turns | empirical (public-neutral) | MRR (private) | **on — keystone; public 0.8911→0.8911** |
| `SINGLE_VALUED_SLOTS` | config.py | (category,size,budget) | DST selective-overwrite: newer positive supersedes on these slots | paper (SOM-DST) | MRR/MTTC | on (correctness, unit-tested) |
| NL pop-suppression | agent.py | pop_weight=0 on NL turns | stop fame burying a well-retrieved target on generic phrases | heuristic | MRR | on (with NL capture) |
| `USE_CE_CONVEX` | config.py | False | score-aware convex CE fusion | empirical (CE-FUSION-01) | MRR | off (see gate) |
| `CE_CONVEX_GATE_MARGIN` | config.py | 0.5 | convex only when belief.margin < this (paraphrase turns) | heuristic | MRR | wired; validating |
| `USE_CATEGORY_GATE` | config.py | False | demote wrong-category lookalikes (confidence-gated) | paper (GenFacet) | MRR/HR | off; needs shadow validation |

Baseline note: clean-cache public RRF = **0.8911** (committed results.json 0.9298 is stale).
