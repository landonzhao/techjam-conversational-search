# Component-by-Component Technical Audit

> Critical audit of every major component against the four competition pillars (I Core Architecture,
> II Dialog Strategy, III Self-Evolution/DCP, IV Evaluation), the five judging criteria (Technical
> Execution, Innovation & Insight, Impact & Relevance, Feasibility & Practicality, Presentation), and
> the four metrics (Hit Rate@10, MRR, MTTC/Efficiency, TechnicalScore). Grounded in
> [ARCHITECTURE.md](ARCHITECTURE.md) and [DECISIONS.md](DECISIONS.md). **No code was changed in this
> pass** — the purpose is to decide, with evidence, what to build next.
>
> **Baseline (results.json, public 200):** HR 0.995 · MRR 0.887 · MTTC 2.695 · Eff 0.8305 ·
> **TechnicalScore 0.9298**. Public is **leak-inflated** (§2 of ARCHITECTURE.md). The honest proxy
> (`evaluator/robustness.py`, `oracle_leakfree.py`) is the real target: **retrieval recall ~99.2%,
> honest hit@10 ~74%, ~97% of honest misses are ranking's fault, median mis-ranked target sits at
> pool rank ~2.** That single finding dominates the prioritization below.

Recommendation classes: **KEEP** (sound, preserve) · **TUNE** (retune values) · **REFACTOR**
(restructure, same behavior) · **REPLACE** (swap the approach) · **EXPERIMENT** (validate a change
behind a flag).

---

## A. Agent orchestrator — `src/agent.py`

**1. Current implementation.** `Agent.respond` sequences the whole turn (§9 ARCHITECTURE). Heavy
objects built once in `__init__`; per-session `ConversationState` in `self._sessions`. ~40 class-flag
toggles form the "flag ledger." No retrieval/ranking/NLU logic lives here.

**2. Strengths.** Clean layering (orchestration owns no algorithms); everything expensive is
init-once; the flag ledger is genuinely excellent engineering discipline (every non-default is
justified with a measurement or a "why-off"); token metering is honest; deterministic default path.

**3. Weaknesses.** `respond` is ~300 lines with deep inline branching — the least single-responsibility
unit; hard to unit-test a single stage in isolation. Candidate flow is untyped (`list[str]` + parallel
score dicts). Some optional branches (structured coverage, price, LTR) add cognitive load for features
that are off.

**4. Competition gap.** Pillar VII orchestration is solid but **not adaptive** — the "dynamic
workflow selection" (Pillar III) is nominal: `OrchestrationPolicy` defaults reproduce the static
pipeline, so no strategy actually changes per turn. Innovation credit here is weak (it *looks*
adaptive but isn't).

**5. Metric impact.** Neutral to all metrics directly; it is the substrate. A refactor into staged
sub-methods would improve testability (Technical Execution) without touching metrics.

**6. Heuristic audit.** No numeric heuristics of its own (delegates to config).

**7. Research.** Adaptive-RAG routing ([Self-RAG](https://www.emergentmind.com/topics/self-rag),
[Retrieval as a Decision: training-free adaptive gating](https://arxiv.org/html/2511.09803)) shows a
small query classifier routing to no-retrieve / single / multi-step. Applicable *conceptually* but our
recall is already 99% — routing retrieval depth buys little; routing *rerank effort* is the useful
analogue (see reranker audit).

**8. Candidate improvements.** (a) **REFACTOR** `respond` into `_understand / _retrieve / _rank /
_decide / _compose` sub-methods (testability, no behavior change). (b) Introduce a lightweight
`Candidate` dataclass to make the stage contract explicit (optional, low value).

**9. Recommendation: REFACTOR (low priority).** Improves Technical-Execution presentation; no metric
movement. Do it opportunistically, not first.

---

## B. Catalog + BM25 — `src/catalog.py`

**1. Current.** JSONL → in-memory dict + SQLite FTS5. `bm25()` = OR-of-terms, field weights
`(0,6,4,2.5,2.5,1.5,1)`, `k1=1.2,b=0.75` (FTS5 defaults), cap 60 terms, `LIMIT pool`.

**2. Strengths.** Zero-dependency, in-memory, fast, reproducible — exemplary for the "lightweight /
no external vector DB" constraint. OR-recall + rank-later separation is the right design.

**3. Weaknesses.** Field weights are heuristic and never swept; title-heavy weighting can favor
keyword-stuffed titles. BM25 `k1/b` never tuned (library defaults). OR-union with 60 terms on long
multi-turn queries can dilute precision (mitigated by ranking).

**4. Competition gap.** Pillar I keyword retrieval fully satisfied. No gap; this is baseline
engineering done well (not innovative, nor needs to be).

**5. Metric impact.** Sets the recall ceiling → **Hit Rate**. Already ~99% recall@200, so upside is
small; downside (a regression) is large. Precision here barely matters (ranking owns it).

**6. Heuristic audit.** `BM25_WEIGHTS` (provisional), `BM25_MAX_TERMS=60` (provisional), `k1/b`
(inherited/untested). Retain weights; a one-off sweep is low-risk but low-reward.

**7. Research.** RRF-vs-hybrid surveys confirm BM25 remains a strong sparse baseline; "sparse excels
at precise entity matching" ([R3AG](https://arxiv.org/html/2604.22849v1)). Nothing suggests replacing
FTS5 BM25 at 50k scale.

**8. Candidate improvements.** (a) **TUNE** field weights + `k1/b` via a recall@200 sweep on the
leak-free set (cheap, but ceiling is ~1 point). (b) Keep as-is otherwise.

**9. Recommendation: KEEP (optionally TUNE).** Not a bottleneck; do not spend early effort here.

---

## C. Dense retrieval — `VectorRetriever`, `src/retrieval.py`

**1. Current.** Precomputed `BAAI/bge-small-en-v1.5` 384-d L2-normalized embeddings; cosine=dot;
query prefix; `argpartition` top-n. `search_decayed` for recency-weighted multi-turn (off, decay=1).

**2. Strengths.** The paraphrase-robust recall track and the semantic term behind the satisfaction
ranker — the single most important honest-set lever. Corpus embedded once; only the query is encoded
live. Fail-safe to BM25.

**3. Weaknesses.** `bge-small` is a 2023 small model; stronger small encoders now exist
(bge/e5/gte-large, or bge-m3 for multi-vector). No ANN (full 50k dot product — fine now, but a
larger model raises encode latency). `search_decayed` is dead code (never enabled).

**4. Competition gap.** Pillar I vector similarity satisfied. Upgrading the encoder is the clearest
"appropriate model use" (Technical Execution) and honest-metric (Impact) lever, but must be measured.

**5. Metric impact.** Drives **Hit Rate** (recall) and, via the satisfaction semantic term, **MRR** on
paraphrase. A better encoder could lift both honest metrics; a worse prefix/model regresses both.

**6. Heuristic audit.** `EMBED_MODEL` (inherited), `SLOT_DECAY=1.0` (arbitrary/unused),
`EMBED_QUERY_PREFIX` (model-mandated, keep).

**7. Research.** MTEB leaderboard-class small encoders (e5-base-v2, gte-base, bge-base-en-v1.5) offer
+2–5 nDCG over bge-small at ~2–3× encode cost; still CPU-viable. [Passage-embedding listwise
reranking](https://arxiv.org/pdf/2406.14848) shows cached embeddings can also drive efficient rerank.

**8. Candidate improvements.** (a) **EXPERIMENT** swap to `bge-base-en-v1.5` (same family, 768-d) and
re-measure honest recall/MRR vs latency (rebuild cache; one flag). (b) Delete or wire `search_decayed`
(decide: use it for override turns or remove it). (c) Consider a stronger checkpoint only if (a) shows
recall headroom — recall is already 99%, so the win is mostly via the *satisfaction semantic term*,
not recall.

**9. Recommendation: EXPERIMENT (medium).** A model swap is a clean, measurable, in-memory test; but
because recall is already saturated, the payoff routes through ranking, so sequence it *after* proving
ranking-side headroom.

---

## D. Fusion (RRF) + intent weight + expansion — `src/retrieval.py`, `_retrieve`

**1. Current.** Weighted RRF `Σ w/(k+rank+1)`, `k=60`; dense weight interpolated
`0.35−0.15·buying_score`; low-weight (0.1) synonym+use-case expansion side-track.

**2. Strengths.** RRF needs no score calibration — robust, standard, correct. Intent-aware weighting
is a genuinely nice touch (buying→BM25, browsing→dense) aligned with the literature ("optimal
retriever is query-dependent"). Expansion is low-blast-radius.

**3. Weaknesses.** **RRF discards score magnitudes** — a decisive dense hit and a marginal one at the
same rank contribute equally. Fixed weights ("static hybridization is often suboptimal" — 2024/25
fusion work). Expansion tables are static/hand-seeded.

**4. Competition gap.** Pillar I fusion satisfied; the "adaptive pipeline" (Pillar III) claim is thin
because weights don't actually adapt beyond the intent interpolation. Convex-combination or learned
fusion would be a defensible Innovation upgrade — but see metric caveat.

**5. Metric impact.** **Hit Rate** primarily (pool composition). Because recall is saturated, fusion
changes mostly reshuffle within the pool → small HR effect, minor MRR effect upstream of the ranker.

**6. Heuristic audit.** `RRF_K=60` (inherited, validated-ish), `*_VECTOR_WEIGHT` (empirical),
`EXPANSION_WEIGHT=0.1` (heuristic). Retain; low priority to tune.

**7. Research.** [RRF vs Convex Combination](https://ceur-ws.org/Vol-4173/T3-7.pdf): CC(α=0.5)
Recall@5 0.726 > RRF(k=60) 0.695 in one study; lower RRF k helps. [Learned/dynamic-α (DAT)] fusion
beats static. **Key insight for us:** the score-magnitude argument matters *more at the rerank fusion
step* (cross-encoder) than at retrieval, because retrieval recall is already maxed.

**8. Candidate improvements.** (a) **TUNE** `RRF_K` downward (30/40) + re-measure recall. (b)
**EXPERIMENT** convex combination at the *rerank* fusion (see reranker audit) rather than retrieval.

**9. Recommendation: KEEP retrieval fusion; move the CC idea to reranking.** The score-calibration
win is real but belongs where magnitudes carry precision — the cross-encoder step.

---

## E. IntentRouter — `src/dialogue.py`

**1. Current.** Lexical sigmoid: `s = 1.5·buy − 1.5·browse + 1.0·hardconstraint + 0.18·(distinct−6)`,
`buying_score=σ(s)`, EMA α=0.6, labels 0.6/0.4. Override via phrase match.

**2. Strengths.** Deterministic, fast, transparent, smoothed. Feeds retrieval mix + personalization +
diversification. Override detection is reliable for the simulator's explicit phrases.

**3. Weaknesses.** **The four coefficients and both cutoffs are hardcoded in `dialogue.py` (not
config) and entirely untested — the largest cluster of arbitrary values in the system.** Wholly
dependent on the simulator's marker phrases ("key requirement", "still exploring"); a free-form real
shopper won't trigger them → intent collapses to the specificity term alone. Binary phrase presence
ignores strength/negation.

**4. Competition gap.** Pillar I routing is *present* but shallow; Pillar II ("intent shift") is
handled only via the explicit override phrase, not via genuine belief-revision. Innovation is weak —
this is a hand-tuned lexicon, not a learned or calibrated router. Impact/Relevance: brittle off the
benchmark distribution.

**5. Metric impact.** Indirect. Wrong intent mis-weights retrieval (small HR/MRR effect) and mis-sets
personalization strength. **MTTC** effect is minor (routing rarely changes the ask). Overall a
low-leverage component metrically, despite being conceptually central.

**6. Heuristic audit.** ±1.5, +1.0, 0.18/pivot-6, 0.6/0.4, EMA 0.6 — all arbitrary/heuristic. Should
be **moved to config and TUNED** against a synthetic intent test set, but expected metric payoff is
small.

**7. Research.** [Learning to Ask: conversational product search via representation
learning](https://arxiv.org/pdf/2411.14466); [RA-Rec semi-structured NL state
tracking](https://arxiv.org/pdf/2406.00033) — LLM/representation intent is stronger but heavier.
[Bayesian inverse preference inference] frames intent shift properly. For our constraints a *calibrated
lexical* router is the right weight class; a learned one risks overfitting 200 sessions.

**8. Candidate improvements.** (a) **TUNE**: lift coefficients to config, sweep on a synthetic
intent/routing set (accuracy + MTTC). (b) Add negation/strength awareness to phrase scoring. (c) Keep
LLM intent *off* (overkill; heavy).

**9. Recommendation: TUNE (low–medium priority).** Fix the arbitrariness (cheap, good for Technical
Execution and judging defensibility) but don't expect much metric movement.

---

## F. State + constraint capture + SlotFiller/NeedModel — `src/dialogue.py`, `src/understanding.py`

**1. Current.** `extract_constraints` lifts verbatim phrases after the simulator marker →
`constraint_phrases` (primary ranking signal). `SlotFiller` regex/vocab → polarity-aware
`Constraint`s; `NeedModel.revise` = value-keyed newer-wins, multi-value coexist. Category via
`CATEGORY_CANON` leftmost-wins.

**2. Strengths.** The raw-phrase channel is the correct design (survives paraphrase, drives coverage &
satisfaction). Non-monotonic revision handles the simulator's override cleanly. Category resolver is
well-tested and thoughtful (canonical buckets, leftmost-wins). Fully deterministic.

**3. Weaknesses.** **Hard contradictions on the same slot/different value are NOT resolved** — "black"
then "actually white" keeps both colors (multi-value coexist); only identical `(slot,value)` triggers
newer-wins. This is a real Pillar II gap ("erase/replace stale constraints"). Regex slot filling is
brittle on paraphrase (hallucinated size, missed reworded material) — acknowledged. `constraint_phrases`
capture is bound to the marker vocabulary; a real shopper's phrasing is not captured as a phrase.

**4. Competition gap.** Pillar II "erasure/replacement of stale constraints" only partially satisfied
(reversal-within-slot missing). "Structured session state" is solid. Innovation: the raw-phrase +
non-monotonic model is a genuinely good insight (above generic slot-filling).

**5. Metric impact.** `constraint_phrases` → **Hit Rate + MRR** (feeds the ranker directly). Correct
stale-constraint handling → **MTTC/MRR** on override/conflict scenarios (30 override + boundary
samples). A same-slot reversal bug could bury the target when a shopper corrects themselves.

**6. Heuristic audit.** Weight 1.0/0.5 (heuristic), polarity window 28 chars / last-3-tokens
(heuristic), brand threshold ≥3 (heuristic), `CatalogVocab` deciles. Mostly reasonable; the polarity
window is the shakiest.

**7. Research.** [RA-Rec / semi-structured NL state tracking (SIGIR
2024)](https://dl.acm.org/doi/10.1145/3626772.3657670); the four-step DST loop (intent → state update
→ action → response) matches our structure. [Situated dynamic/implicit preference
reasoning](https://arxiv.org/pdf/2604.20749) motivates real reversal handling.

**8. Candidate improvements.** (a) **EXPERIMENT** same-slot reversal: on a fresh positive constraint
for a single-valued slot (color/size/material/category), demote or drop prior conflicting values
(guard multi-value slots like feature). Deterministic, testable, targets override/boundary MRR. (b)
**TUNE** polarity window. (c) Keep LLM slots off (measured net-negative).

**9. Recommendation: EXPERIMENT (medium).** The reversal fix is a small, deterministic, well-scoped
Pillar-II win with a clean synthetic test — a strong secondary target.

---

## G. CoverageReranker — `src/ranking.py` (fallback default)

**1. Current.** Verbatim IDF coverage of raw phrases + full-phrase/prefix bonuses + pop blend/tie-break
+ optional floors/gates/semantic/structured/price tracks. Now the *fallback* (satisfaction is default).

**2. Strengths.** Exploits the leak optimally → the ~0.93 public number. The retrieval-floor and
discrimination-gate work is genuinely sophisticated and honestly measured. Returned score is always
raw coverage (belief stays clean).

**3. Weaknesses.** Only wins on the leak; collapses to popularity on paraphrase (the honest failure
mode). Huge parameter surface (a dozen knobs) — maintenance and overfitting risk. Much of it is now
superseded by the satisfaction ranker.

**4. Competition gap.** Pillar I semantic ranking is satisfied *only on the leaky distribution*.
Impact/Relevance is weak in isolation (real shoppers paraphrase). Its value now is as a leaky-set
guardrail and an ablation baseline.

**5. Metric impact.** On public: dominant **MRR/HR**. On honest: near-zero to negative unless the
floor/gate fire. Keeping it as fallback protects the public guardrail.

**6. Heuristic audit.** See DECISIONS §4 — many knobs, several off/negative (semantic coverage −0.045).
Retain the validated ones (pop_blend 0.1, retrieval_weight 1.0, prefix tier); the off tracks are
research residue.

**7. Research.** Verbatim/IDF coverage ≈ classic exact-match features in LTR; the honest fix is
semantic matching (already the satisfaction ranker).

**8. Candidate improvements.** (a) **KEEP** as the leaky-guardrail fallback. (b) **REFACTOR** later:
prune the off-by-default tracks into an experiments module to shrink the surface. (c) Do not invest in
new coverage knobs.

**9. Recommendation: KEEP (as fallback), REFACTOR later.** Frozen; the action is in the satisfaction
+ rerank path.

---

## H. NeedSatisfactionScorer — `src/ranking.py` (default ranker)

**1. Current.** `match = max(lexical_IDF_fraction, α·cos)`, α=1.0; length-weighted mean; adaptive
popularity `w_pop=0.15·(1−specificity)`. Coverage is the α=0,pop=0 special case. Ties keep retrieval
order.

**2. Strengths.** The best idea in the repo: a clean superset of coverage that degrades gracefully to
the leaky behavior while adding paraphrase robustness. Validated (+35% pillar_free) for −0.014 public.
Reuses cached text/IDF/embeddings (no extra cost). Adaptive popularity is principled (fame fades with
specificity).

**3. Weaknesses.** The `max(lexical, semantic)` combiner is heuristic — a strong lexical partial can
mask a better semantic match and vice versa; magnitudes aren't calibrated. α, pop_weight, specificity
ref are hand-set. Semantic term inherits encoder quality (bge-small). No hard-constraint enforcement:
a wrong-category product with high semantic similarity can still rank high.

**4. Competition gap.** Pillar I semantic ranking well satisfied and genuinely innovative (defensible
vs generic RAG). Gap: no *conflict/hard-constraint* handling (planned 3-state SATISFIED/CONFLICT/
UNKNOWN, RANKING_REDESIGN) — the honest set still mis-ranks category/material violators.

**5. Metric impact.** The primary **MRR** driver on the honest/private distribution and a **Hit Rate**
contributor. This is *the* highest-leverage metric surface (97% of honest misses are ranking).

**6. Heuristic audit.** α=1.0, pop_weight=0.15, specificity_ref=3 (all empirical/heuristic, validated
region). The `max` combiner design itself is untested vs alternatives (weighted sum, calibrated).

**7. Research.** [Convex combination beats RRF](https://ceur-ws.org/Vol-4173/T3-7.pdf) when scores are
calibrated; cross-encoder precision ([Set-Encoder](https://arxiv.org/pdf/2404.06912),
[FIRST](https://aclanthology.org/2024.emnlp-main.491/)) resolves rank-2 lookalikes far better than
bi-encoder cosine. Hard-constraint gating ≈ faceted filtering in product search.

**8. Candidate improvements.** (a) **EXPERIMENT** a hard-constraint/category consistency gate before
final ranking (demote violators when the constraint is confidently extracted). (b) **TUNE** α/pop via
the pop-ablated eval_matrix. (c) **EXPERIMENT** replacing `max` with a small calibrated combination —
but the bigger precision lever is the cross-encoder (component I).

**9. Recommendation: KEEP + EXPERIMENT (high).** Sound core; the headroom is in *what feeds/gates it*
(hard-constraint gate) and the *precision reranker* stacked on it.

---

## I. Reranking stack — CrossEncoder / LLM / LTR — `src/reranker.py`, `src/ranking_features.py`

**1. Current.** `CrossEncoderReranker` (`ms-marco-MiniLM-L-6-v2`, **ON**, depth 50, RRF weight 1.0,
near-tie margin 0 = always) fused into the order via RRF. `LLMReranker` (Gemini listwise, off).
`LTRModel` (linear, 9 features, off, "not shown to beat default").

**2. Strengths.** Cross-encoder is the single largest measured honest win (pillar_free 0.46→0.66, MRR
0.31→0.59) at $0/offline. Everything fails safe to input order. LTR is designed to *bound* the leak
feature — good instinct.

**3. Weaknesses.** **The CE is fused via RRF at fixed weight 1.0 — discarding its score magnitudes,
which are exactly the precision signal that separates rank-2 lookalikes.** Depth 50 / weight 1.0 /
always-on are untuned. `ms-marco-MiniLM-L-6-v2` is a small, MS-MARCO-domain (not e-commerce) model.
LTR risks overfitting 200 public sessions (few sessions, distinct private users — the research warns
GBDT/LTR overfit on small data).

**4. Competition gap.** Pillar I "LLM/semantic reranking" is satisfied by the CE; the *LLM* reranker
is off, so the flashy "LLM ranking" story is currently deterministic-CE. Innovation: strong (a
retrieve→bi-encoder→cross-encoder cascade is textbook-correct and defensible).

**5. Metric impact.** **MRR** (and HR when it lifts a rank-11 target to ≤10) on the honest/private
distribution — the metric that most needs help. Naive RRF fusion is likely leaving MRR on the table.

**6. Heuristic audit.** `CE_DEPTH=50`, `CE_WEIGHT=1.0`, `RERANK_NEAR_TIE_MARGIN=0` (all provisional,
untuned); CE model choice (inherited). LTR feature set (reasonable) + standardization (fine).

**7. Research.** [RRF vs Convex Combination](https://ceur-ws.org/Vol-4173/T3-7.pdf) (calibrated CC >
RRF); [FIRST single-token listwise, +50% speed](https://aclanthology.org/2024.emnlp-main.491/);
[Set-Encoder 85× faster than RankGPT](https://arxiv.org/pdf/2404.06912); [Efficiency-effectiveness
rerank FLOPs](https://arxiv.org/pdf/2507.06223). Consensus: **cross-encoder precision >> bi-encoder
cosine for top-k ordering**, and calibrated score fusion > rank fusion when magnitudes are meaningful.

**8. Candidate improvements.** (a) **EXPERIMENT** replace the CE's RRF fusion with a **calibrated
convex combination** of min-max-normalized satisfaction + CE scores (`s = (1−β)·sat_norm + β·ce_norm`),
sweeping β — directly uses the score magnitudes RRF throws away. (b) **TUNE** `CE_DEPTH` /
near-tie-gating (only fire CE when the top is contested — efficiency). (c) **EXPERIMENT** a stronger /
e-commerce cross-encoder checkpoint (e.g. `bge-reranker-base`) if (a) shows headroom.

**9. Recommendation: EXPERIMENT (highest).** The cross-encoder is our best honest-metric asset and is
currently integrated naively; calibrated fusion is lightweight, measurable, and low-risk (CE already
on). **This is the recommended first component (see §Roadmap).**

---

## J. BeliefModel + converge — `src/understanding.py`

**1. Current.** Over top-20: margin, normalized entropy, stability →
`item_conf = 0.5·margin + 0.3·(1−entropy) + 0.2·min(stable/2,1)`; per-slot uncertainty →
`need_conf`; `confidence = min(item,need)`. `converge`: ≥0.60 DELIVER / ≥0.35 CONFIRM / else PROBE.

**2. Strengths.** An explicit, inspectable belief is a genuine Pillar II/III asset (most agents don't
model their own confidence). Drives clarification and the validated adaptive reveal. Cheap arithmetic.

**3. Weaknesses.** The 0.5/0.3/0.2 blend is hardcoded (not config) and **uncalibrated** — confidence
is not validated against realized hit probability, yet it gates reveal (which is worth +0.033). No
unit test. `min(item,need)` can be dominated by an over-pessimistic need term.

**4. Competition gap.** Pillar III self-assessment: present and above baseline. Gap: calibration —
"confidence" is an unvalidated heuristic, so the reveal gate could be firing sub-optimally.

**5. Metric impact.** Indirect but real via reveal → **MRR** and via converge → **MTTC**. Mis-calibrated
confidence either reveals too early (bad MRR lock) or too late (wasted turn).

**6. Heuristic audit.** 0.5/0.3/0.2 blend, TOPN 20, attr floor 0.5, CONVERGE 0.60/0.35 — all
heuristic/arbitrary. The blend and CONVERGE thresholds should be **calibrated** against logged
first-hit data.

**7. Research.** Confidence calibration for ranking / selective prediction; EVPI-style question value
([Rao & Daumé, "Learning to Ask Good Questions"] via the clarification search). A calibrated
`P(target in top-k)` would make reveal principled.

**8. Candidate improvements.** (a) **TUNE/EXPERIMENT** calibrate confidence: fit the reveal threshold
against logged traces (the tracer already records target rank per turn). (b) Move the blend to config.
(c) Add a belief unit test.

**9. Recommendation: TUNE (medium).** Calibrating the reveal gate could sharpen MRR further, but it is
downstream of the ranker — sequence after the reranker win.

---

## K. Clarification — QuestionSelector + next_ask — `src/understanding.py`, `src/dialogue.py`

**1. Current.** `select` picks `argmax uncertainty·DECISION_WEIGHT·guidance`, with CONFIRM/comparison
special cases and (off) adaptive-clarify feature facets. **But default `INFO_GAIN_MODE="display"`
makes the scored `ask_attribute` almost always `"other"`** — the info-gain choice appears only in the
(unscored) message.

**2. Strengths.** The selector itself is sophisticated (pool-derived facets, comparison questions,
distinctive-attribute confirm). `"other"` maximizes verbatim disclosure — a smart benchmark
adaptation given the leak.

**3. Weaknesses.** **The clever selection is off the scored path.** So the genuine Pillar-II
clarification capability is under-demonstrated where it counts. Tension: asking specific attributes
reduces the verbatim leak the public score depends on — so `"other"` is *rational for the public
metric* but *poor for real MTTC/UX and for judging Innovation*. `DECISION_WEIGHT` and the 0.15
comparison margin are arbitrary. No unit test.

**4. Competition gap.** Pillar II proactive clarification: **only partially demonstrated** (the scored
behavior is "ask other"). Innovation credit is at risk — a judge inspecting `ask_attribute` sees
"other" every turn. Presentation gap: our best dialog logic is invisible to the metric.

**5. Metric impact.** **MTTC/Efficiency.** Current MTTC 2.695 (Eff 0.83) is already good, so headroom
is limited: MTTC 2.7→2.0 ≈ Eff 0.83→0.90 ≈ **+0.014 TechnicalScore**. Switching to real ask-mode risks
the public leak (specific asks → less verbatim disclosure → weaker coverage) — a **public-guardrail
risk**.

**6. Heuristic audit.** `DECISION_WEIGHT` map, comparison margin 0.15, `ASK_PRIORITY`, phase
thresholds 6/7 — heuristic/arbitrary.

**7. Research.** [Facet-driven clarifying questions (SIGIR
2021)](https://dl.acm.org/doi/10.1145/3471158.3472257); [EVPI "Learning to Ask Good
Questions"](https://dl.acm.org/doi/10.1145/3527546.3527578); [Ask-or-Recommend empirical study (CIKM
2024)](https://dl.acm.org/doi/10.1145/3627673.3679875) — ask early, ask facet-discriminating
questions. Our `_feature_facet` is already facet-driven; the gap is *activating* it on the scored path
without losing the leak.

**8. Candidate improvements.** (a) **EXPERIMENT** a *hybrid* ask policy: keep `"other"` when coverage
is winning (verbatim turns) but switch to a discriminating facet ask when the belief is genuinely
uncertain AND coverage is uninformative (paraphrase) — measured on both public (guardrail) and
robustness (MTTC). (b) **TUNE** DECISION_WEIGHT/margin. (c) Move phase thresholds to justified values.

**9. Recommendation: EXPERIMENT (medium, later).** Real Innovation/Pillar-II upside, but limited
metric headroom and genuine public-guardrail risk — not first.

---

## L. Adaptive reveal — `_reveal_count`, `src/agent.py`

**1. Current.** Hold back to K=1 while `confidence<0.55` and `turn<4`; always reveal at confidence,
turn cap 4, or last turn. Validated +0.033 public (MRR 0.705→0.861).

**2. Strengths.** A clever, validated exploitation of the evaluator's first-appearance MRR rule.
Honestly documented. Cheap.

**3. Weaknesses.** Pure benchmark-mechanic optimization — improves *measured* MRR more than UX;
depends on the uncalibrated belief confidence (component J). A judge may read it as gaming. Fragile if
the private evaluator changes the first-appearance rule.

**4. Competition gap.** Pillar IV metric-awareness: fully exploited. Innovation: borderline (mechanic
gaming vs insight). Impact/Relevance: low (not a real feature).

**5. Metric impact.** Direct **MRR** (large, validated) and **MTTC** (holding back delays first hit —
a real trade managed by the turn cap).

**6. Heuristic audit.** REVEAL_CONFIDENCE 0.55, K 1, turn cap 4 — empirical/validated. Retain.

**7. Research.** N/A (evaluator-specific). Related: selective prediction / risk-coverage tradeoffs.

**8. Candidate improvements.** (a) **KEEP.** (b) Tie to a *calibrated* confidence (component J) so the
hold-back fires on true rank probability. (c) Stress-test robustness to rule changes.

**9. Recommendation: KEEP.** Validated; only revisit after calibrating belief.

---

## M. Personalizer + DCP context engine — `src/ranking.py`, `src/context_engine.py`

**1. Current.** Personalizer (pop + tag pre-sort, skipped when satisfaction on). DCP: ContextDistiller
(decay 0.9, prune), ProfileService (dormant in eval), OrchestrationPolicy (defaults = static),
GuidanceLearner (online slot reweighting).

**2. Strengths.** Clean, best-effort, fail-safe, score-neutral. Long-term profile + guidance learning
are the concrete Pillar-III artifacts. Honestly labeled UNPROVEN.

**3. Weaknesses.** **The entire DCP layer is on but score-neutral and unproven** — it demonstrates
Pillar III terminology (self-evolution, memory) without measured metric impact. Profiles are dormant
(distinct users). OrchestrationPolicy is adaptive in name only (defaults reproduce static). Flat
popularity (Personalizer) is corrosive on honest data (hence disabled under satisfaction).

**4. Competition gap.** Pillar III is the **weakest-demonstrated pillar**: the capability exists but
can't be shown to move the metric, and the official eval structurally can't exercise long-term
profiles. Innovation claim is fragile under scrutiny ("is this adaptive or just labeled adaptive?").

**5. Metric impact.** ~Zero by design. Risk: reviewers discount Pillar III if we can't show an
ablation delta.

**6. Heuristic audit.** DECAY 0.9, MAX/MIN/floor 12/4/0.15, EMA 0.6, halflife 45, λ/EMA 0.5/0.3 — all
heuristic. Fine (score-neutral).

**7. Research.** [Cognis context-aware memory](https://arxiv.org/pdf/2604.19771); [multi-subsession
conversational rec](https://arxiv.org/pdf/2310.13365); session-based recommendation. To *earn* Pillar
III credit we need a setting where memory measurably helps (a returning-user synthetic set).

**8. Candidate improvements.** (a) **EXPERIMENT (WS4 ablation)**: build a returning-user synthetic set
and show DCP profiles lift it — converts "labeled adaptive" into "measured adaptive." (b) Make
OrchestrationPolicy actually change one decision (e.g. rerank depth by phase) and measure. (c)
Otherwise **KEEP** score-neutral.

**9. Recommendation: EXPERIMENT (to defend Pillar III), else KEEP.** Presentation/Innovation value,
not a metric lever.

---

## N. LLM layer + GeminiClientPool — `src/llm_inference.py`, `src/reranker.py`, `src/keys.py`

**1. Current.** Slot extraction / use-case / response / rerank on Gemini `flash-lite`, all off/inert,
schema-constrained, cached, metered, fail-safe. Key pool with rotation.

**2. Strengths.** Textbook isolation of provider code; honest metering; schema-enum outputs; every
path degrades to deterministic. Measured LLM slots net-negative and correctly kept off.

**3. Weaknesses.** Currently contributes **nothing** to the scored path — the "LLM-powered" story is
aspirational on the metric. `query_rewrite.txt` unused; `LLMReranker` prompt inline (inconsistent with
`prompts/`). Over-reliance risk is *avoided* (good), but so is any LLM upside.

**4. Competition gap.** "Appropriate model/API use" (Technical Execution) is demonstrated
structurally; but no LLM component earns a metric. If judges weight LLM usage, we under-show it.

**5. Metric impact.** Zero when off. The only plausible positive is the LLM *reranker* on genuinely
paraphrased near-ties — but rate limits + latency + token cost make it a robustness hook, not a
leaderboard lever.

**6. Heuristic audit.** LLM_SLOT_MAX_REGEX 2, temps 0/0.3, depths — provisional.

**7. Research.** [FIRST](https://aclanthology.org/2024.emnlp-main.491/) / [Set-Encoder](https://arxiv.org/pdf/2404.06912)
make LLM listwise rerank cheaper; but our local cross-encoder already captures most of the precision at
$0. LLM rerank is dominated by CE for our constraints.

**8. Candidate improvements.** (a) **KEEP** off; wire `query_rewrite` or delete it (tidy). (b) Move the
LLMReranker prompt to `prompts/`. (c) Only revisit LLM rerank if the CE + calibrated fusion plateaus.

**9. Recommendation: KEEP (off), minor REFACTOR (tidy prompts).** Not a metric lever now.

---

## Repository-wide prioritization

**Expected Value ≈ (Expected Metric Gain × Confidence) / Implementation Cost** (heuristic, not
official). Gain estimates are on the **honest/private proxy** unless noted, because that is where
headroom lives (public is near-ceiling at 0.93).

| # | Component | Problem | Proposed improvement | Expected metric | Confidence | Cost | Risk | EV |
|---|---|---|---|---|---|---|---|---|
| 1 | **Reranker fusion (I)** | CE fused via RRF, magnitudes discarded; fixed weight/depth | Calibrated convex-combination of satisfaction+CE scores; sweep β/depth | **MRR↑ (honest), HR↑ marginal** | Med-High | **Low** | Low (CE already on; guardrail-checked) | **High** |
| 2 | Hard-constraint gate (H) | wrong-category/material lookalikes rank high on honest set | demote candidates violating a confidently-extracted hard constraint before final rank | MRR↑, HR↑ (honest) | Medium | Low-Med | Med (public leak; gate on confidence) | High |
| 3 | Same-slot reversal (F) | "black→white" keeps both; stale constraint not erased | value-reversal for single-valued slots | MRR/MTTC↑ on override+boundary (40 samples) | Medium | Low | Low | Med-High |
| 4 | Dense encoder (C) | bge-small is dated; semantic term is honest lever | swap to bge-base / stronger reranker checkpoint; re-measure | HR/MRR↑ (honest) | Medium | Med (rebuild cache) | Low | Med |
| 5 | Intent coeffs (E) | 6 arbitrary hardcoded values, untested | lift to config + sweep on synthetic intent set | MTTC↑ small; Technical-Execution defensibility | Low-Med | Low | Low | Med |
| 6 | Belief calibration (J) | reveal gate uses uncalibrated confidence | calibrate threshold from tracer logs | MRR↑ small | Low-Med | Low-Med | Low | Med |
| 7 | Clarification ask-mode (K) | best dialog logic off scored path | hybrid facet-ask on uninformative turns | MTTC↑ small; Innovation | Low-Med | Med | **Med-High** (public leak) | Low-Med |
| 8 | DCP ablation (M) | Pillar III unproven | returning-user synthetic set + ablation | Presentation/Innovation | Medium | Med | Low | Low-Med |

### Top 5 by expected value

1. **Reranker fusion — calibrated convex combination (component I).** Highest EV: our best honest
   asset (cross-encoder) is integrated naively; a calibrated fusion is cheap, measurable, low-risk.
2. **Hard-constraint / category gate (component H).** Directly attacks the honest failure mode
   (rank-2 wrong-category lookalikes); deterministic and explainable; gate on extraction confidence to
   protect public.
3. **Same-slot reversal in NeedModel (component F).** Small deterministic Pillar-II correctness fix
   with a clean synthetic test; helps the 40 override/boundary samples.
4. **Dense encoder upgrade (component C).** Sequenced after 1–2 confirm ranking headroom; clean
   in-memory swap, but recall is saturated so payoff routes through the semantic term.
5. **Intent-router de-arbitrariness (component E).** Cheap defensibility win (move to config + sweep);
   modest metric payoff but removes the biggest cluster of untested magic numbers.

---

## Recommended FIRST component: **Reranker fusion — calibrated convex combination (component I)**

**Hypothesis.** Replacing the cross-encoder's rank-only RRF fusion with a **calibrated convex
combination of normalized satisfaction and cross-encoder scores** will raise honest-set MRR (and
marginally Hit@10) **without regressing the public guardrail**, because RRF discards the
cross-encoder's score *magnitudes* — precisely the precision signal that separates the correct target
from the rank-2 lookalikes the honest set fails on.

**Current baseline behavior.** `Agent.respond` (lines 481–494): the satisfaction ranker orders the
pool; then, unconditionally (near-tie margin 0), the cross-encoder scores the top `CE_DEPTH=50` and its
**order** is RRF-fused at `CE_WEIGHT=1.0` (`rrf(base, ce_order, 1.0)`). Only the *rank* of each CE
result enters fusion; the raw relevance score is thrown away. `ce_score_map` is computed but only used
(when on) by the off-by-default LTR.

**Proposed change (design only — not implemented this pass).** For the top-`CE_DEPTH` candidates,
compute a blended score
```
score(c) = (1 − β)·norm(satisfaction(c)) + β·norm(ce_score(c))          β ∈ [0,1]
norm(·)  = min-max over the CE-scored head (fallback: rank-based if degenerate)
```
re-sort the head by `score`, keep the tail as-is. `β=1` ≈ CE-only, `β=0` ≈ satisfaction-only; sweep β.
Keep a flag `USE_CE_CONVEX` and `CE_BETA` in `config.py`; the current RRF path stays as the fallback.
Fail-safe: if CE returns nothing or scores are degenerate, use the existing order.

**Why first.** (a) The repo's own oracle says **97% of honest misses are ranking** with the target at
median rank ~2 — precision reranking is the exact remedy. (b) The cross-encoder already delivers the
largest measured honest win (pillar_free 0.46→0.66) but is used naively. (c) Lowest cost (≈30 lines,
one flag), fully in-memory/offline/$0, deterministic, and reversible. (d) Research consensus:
calibrated convex combination > RRF when score magnitudes are meaningful, and cross-encoder precision
>> bi-encoder cosine for top-k ordering.

**Relevant research.**
- [RRF vs Convex Combination (CEUR)](https://ceur-ws.org/Vol-4173/T3-7.pdf) — CC(α) can beat RRF when
  scores are calibrated; motivates the convex form.
- [Set-Encoder (arXiv 2404.06912)](https://arxiv.org/pdf/2404.06912),
  [FIRST (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.491/) — cross-encoder / listwise
  precision resolves near-duplicate top-k better than bi-encoders; efficiency techniques if we later
  deepen.
- [Efficiency-effectiveness rerank FLOPs (arXiv 2507.06223)](https://arxiv.org/pdf/2507.06223) — guides
  the depth/gating sweep.

**Synthetic test design.**
- **Ranking-isolation harness (primary):** reuse `scripts/oracle_leakfree.py` / `eval_matrix.py`. Build
  a fixed set of (query, candidate-pool-containing-target) cases from `data/language_stress_set.jsonl`
  and `data/pillar_free.jsonl`; measure target rank **before vs after** the fusion change (the tracer
  already logs `rank_in_pool`).
- **Near-duplicate stressor (new, small):** hand-pick ~30 pools where the target and 2–3 lookalikes
  differ only on a distinctive feature/model/silhouette (the case CE should win and cosine loses).
- **Guardrail:** the official evaluator on `public_set.jsonl` (must not regress).
- **Honest:** `evaluator/robustness.py` paraphrase mode.

**Metrics to collect.** Per set: Hit@10, MRR, and (from traces) median target `rank_in_pool` and
Δrank; β-sweep curve; per-turn latency (CE encode already paid, so ≈ neutral).

**Acceptance criteria.**
- Leak-free / pillar_free **MRR improves ≥ +0.01** (ideally toward the CE's demonstrated ceiling) at
  the chosen β, with **Hit@10 non-decreasing**.
- Public `recommended_technical_score` **regresses ≤ 0.003** (within the deliberate-tradeoff band).
- Median honest target `rank_in_pool` decreases.

**Regression criteria (reject/rollback).**
- Public TechnicalScore drops > 0.005, **or** honest MRR fails to beat the current RRF fusion at any
  β, **or** latency per turn rises materially (it should not — CE scores are already computed).

**Implementation plan (when approved).**
1. Add `USE_CE_CONVEX` + `CE_BETA` to `config.py` (+ flag-ledger entry) and DECISIONS.md rows.
2. In `Agent.respond`, when CE scores exist, compute the min-max-normalized convex blend over the head
   and re-sort; keep the RRF path as the `USE_CE_CONVEX=False` fallback. No change to the returned
   score map (belief still sees raw satisfaction).
3. Unit test in `tests/test_components.py`: a stub CE score map must reorder a lexically-tied head by
   the blend; degenerate scores must preserve order.
4. Sweep β ∈ {0.3,0.5,0.7,1.0} with `scripts/eval_matrix.py` (relevance-vs-fame column) + robustness +
   public; record in an experiment log (date, commit, config, deltas).
5. Ship on only if acceptance criteria hold; otherwise keep off behind the flag with the measured
   result documented.

**Do NOT touch:** retrieval, session state, the Agent API, the evaluator, or the returned
`recommendations`/`usage` shape. Invariant: output remains a permutation of the input pool; the belief
score map stays raw satisfaction.
