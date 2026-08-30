# Experiment Log

Chronological record of measured experiments. Each entry: hypothesis → method → results → decision.
Newest first. Decisions: **SHIP** · **PROMISING—ITERATE** · **NO DIFFERENCE** · **REJECT**.

---

## EXP CONSOLIDATION-03 — Merge teammate branch-ranking + audit fixes

**Date:** 2026-08-31 · **Decision: adopt her MECHANISM, revert her VALUES.**

### What was merged
`origin/branch-ranking` (commit 1ed0524) — a multi-channel popularity prior for
`NeedSatisfactionScorer`: `_adaptive_prior` fuses a popularity channel and an average-rating quality
channel, weighted per-candidate by `_sem_gate(sem_conf)` (an already-confident semantic match is not
overwritten by a more popular near-neighbour); plus `refresh_satisfaction_scorer()` so sweep
overrides propagate; plus tuned defaults `POP_WEIGHT 0.15→0.3`, `SPECIFICITY_REF 3→6`. Hand-merged
into our working tree (my NL-turn `pop_weight=0` threads through `_adaptive_prior(pop_weight=)`).
57 tests pass.

### Measurement (clean caches, LLM off, full sets)

| config | public Tech | pillar_free Tech | pillar_free MRR |
|---|---|---|---|
| my-fixes + old flat prior (pre-merge) | 0.8911 | — | (iso RRF 0.452) |
| merged w/ her values (pop 0.3, ref 6) | **0.8629 ❌ below 0.88 floor** | 0.6388 | 0.541 |
| **her mechanism + reverted values (pop 0.15, ref 3)** | **0.8842 ✅** | **0.6549** | **0.558** |

### Finding
Her tuned VALUES (0.3 / 6), measured +0.036 on a 25-row subset, **did not transfer** — on our full
pipeline they dropped public to 0.8629, below the 0.88 floor. Her MECHANISM (multi-channel + semantic
gate) with the proven values (0.15 / 3) is **strictly better on both axes** (public 0.8842,
pillar_free 0.6549 vs 0.6388): the semantic gate — not the aggressive values — is what protects the
honest long-tail. Adopted mechanism + reverted values as the consolidated default; her aggressive
values reverted with a note to re-sweep on full sets, not subsets.

### Consolidated default (shipped)
Her `_adaptive_prior`/`_sem_gate`/`refresh_satisfaction_scorer` + `POP_WEIGHT=0.15`,
`SPECIFICITY_REF=3` + my FIXES-02 (NL capture, DST revision, config hygiene; gated convex & category
gate OFF). Public **0.8842** (above floor), pillar_free **0.6549**. Gated convex remains the honest
lever (isolation +0.06/+0.096 MRR; end-to-end pre-merge +0.052 leak-free for −0.007 public) — off
until its margin is tuned to hold the floor on this new base.

---

## EXP FIXES-02 — Audit build sprint (keystone NL capture + DST revision + gates + config)

**Date:** 2026-08-30 · **Components:** constraint capture, NeedModel, ranking fusion, config.

### Keystone finding — ranking is coupled to the evaluator's disclosure syntax
`extract_constraints` only fires on the simulator's `"key requirement is:"` marker (verified:
returns `[]` for every natural-language turn). So on real shopper language `constraint_phrases` is
empty → the default satisfaction/coverage ranker **no-ops and falls back to raw retrieval order**.
This is *the* reason the shadow suite (natural language) scored MRR 0.20 while the marker-carrying
honest sets looked healthier — our primary ranking signal doesn't exist on natural language.

### Fixes built (research-grounded)
1. **NL constraint capture** (`USE_NL_CONSTRAINTS`, default **ON**). When a turn carries no marker,
   the ranker consumes the NeedModel's positive slot values as phrases (`Agent._nl_rank_phrases`), so
   it fires on natural language. On NL turns, popularity is suppressed (`rank(..., pop_weight=0)`) so
   generic regex phrases don't let fame bury a well-retrieved unpopular target. **Public: 0.8911 →
   0.8911 (identical, neutral)** — NL turns on public (browsing turn 1) are held back by adaptive
   reveal anyway, and marker turns are guarded off. Safe default-on; the keystone for the private set.
   Research: query-understanding → structured constraints → constrained ranking (GenFacet; relevance
   filtering). *Caveat:* regex extraction is generic (e.g. "sweater","wool"), so this helps
   category-driven cases, not pure semantic-feature cases (those need deeper NL extraction / dense).
2. **DST single-valued slot revision** (`SINGLE_VALUED_SLOTS={category,size,budget}`, default ON).
   `NeedModel.revise` now overwrites older positives of a single-valued slot ("ankle boots" →
   "block-heel sandals" leaves category=sandal). Multi-valued slots (color/material/…) still coexist.
   Research: SOM-DST selective overwrite / mentioned-slot pools. Unit-tested; roadmap #3.
3. **Gated convex CE fusion** (`CE_CONVEX_GATE_MARGIN`, default OFF/wired). CE-FUSION-01 follow-up:
   convex only on low-margin (paraphrase) turns; RRF on confident verbatim turns (protect public).
   Validation in progress.
4. **Hard-constraint category gate** (`USE_CATEGORY_GATE`, default OFF). `apply_category_gate` demotes
   candidates whose own title resolves to a different category than the known need category.
   Non-destructive, confidence-gated. Roadmap #2; needs shadow-suite validation before default-on.
5. **Config hygiene** (roadmap #5). Moved the previously hardcoded IntentRouter coefficients
   (1.5/1.5/1.0/0.18, cutoffs 0.6/0.4), BeliefModel item-confidence blend (0.5/0.3/0.2), and the
   QuestionSelector comparison margin (0.15) into `config.py`. Zero behavior change; removes the
   largest cluster of untested magic numbers (DECISIONS §12). Values unchanged → all tests pass.

### Shadow teaser (10 sessions) — why it can't yet judge the gates
All flag combinations net MRR 0.10–0.20 on 10 sessions; changes trade individual cases (s3↔s10).
**10 sessions cannot resolve ranking-fusion changes** — this is the concrete evidence that the full
shadow suite must be built before gated-convex / category-gate defaults can be set. The keystone NL
fix and DST revision are shipped on their own merits (public-neutral / correctness + unit tests).

### Note on baseline
Clean-cache public baseline (RRF, LLM off) = **0.8911**, confirming the committed `results.json`
(0.9298) predates the current working tree and is stale. Use 0.8911 as the working public baseline.

---

## EXP CE-FUSION-01 — Calibrated convex fusion of satisfaction + cross-encoder

**Date:** 2026-08-30 · **Component:** reranker fusion (roadmap #1) · **Decision: PROMISING — ITERATE
(kept OFF by default).**

### Hypothesis
The cross-encoder's *score magnitudes* are discarded by the current rank-only RRF fusion
(`agent.py:490–494`: `ce_order` → `rrf(base, reordered_head, CE_WEIGHT)`; `ce_score_map` is consumed
only by the off-by-default LTR). Replacing RRF with a **calibrated convex combination**
`FinalScore = (1−β)·SatNorm + β·CENorm` (min-max over the CE head) should lift honest-set MRR by using
the precision signal that separates rank-2 lookalikes, without regressing the public guardrail.

### Verification of the premise
Confirmed at the code level: CE computes real scores into `ce_score_map`, but fusion uses only their
rank order. Premise TRUE.

### Method
- **Ranking-isolation harness** `scripts/exp_ce_fusion.py`: replays the official disclosure loop on
  honest sets; captures the exact pre-CE satisfaction pool + per-candidate satisfaction score + CE
  score per turn (by wrapping `CrossEncoderReranker.scores` and `NeedSatisfactionScorer.rank`); then
  re-ranks the **identical** captured pool under every strategy. Retrieval is held fixed, so any delta
  is fusion-attributable. Metric: first-top-10-appearance MRR/Hit@10 across turns (full pool, no
  reveal hold-back) + best-rank distribution + head-to-head improved/worsened.
- **Public guardrail:** official `evaluator.local_evaluator.evaluate` on `public_set.jsonl`, RRF vs
  convex β∈{0.5,0.6,0.8}, LLM off (matches shipped scoring config).
- Implementation: `convex_fuse()` in `src/retrieval.py` behind flags `USE_CE_CONVEX` / `CE_BETA`; RRF
  path preserved as the fallback. Unit tests: `ConvexFuseTest` (5 cases).

### Baseline & sweep (ranking-isolation, honest sets)

`language_stress_set` (250 sessions, 247 target-in-pool):

| strategy | MRR | Hit@10 | Hit@1 | medRank | meanRank | improved/=/worsened vs RRF |
|---|---|---|---|---|---|---|
| **rrf (baseline)** | 0.6331 | 0.879 | 0.628 | 1.0 | 8.0 | 0/247/0 |
| sat (order only) | 0.4913 | 0.798 | 0.441 | 2.0 | 11.4 | 11/125/111 |
| β=0.2 | 0.5831 | 0.858 | 0.538 | 1.0 | 8.8 | 29/151/67 |
| β=0.4 | 0.6399 | 0.895 | 0.611 | 1.0 | 7.8 | 36/172/39 |
| β=0.5 | 0.6757 | 0.907 | 0.660 | 1.0 | 7.5 | 44/182/21 |
| **β=0.6** | **0.6929** | 0.907 | 0.700 | 1.0 | 7.3 | 53/178/16 |
| β=0.8 | 0.6929 | 0.915 | 0.725 | 1.0 | 7.3 | 58/174/15 |
| β=1.0 | 0.6759 | 0.911 | 0.684 | 1.0 | 7.5 | 57/161/29 |

`pillar_free` (240 sessions, 237 in-pool):

| strategy | MRR | Hit@10 | Hit@1 | medRank | improved/=/worsened |
|---|---|---|---|---|---|
| **rrf (baseline)** | 0.4521 | 0.781 | 0.456 | 2.0 | 0/237/0 |
| β=0.4 | 0.4485 | 0.810 | 0.422 | 2.0 | 51/144/42 |
| β=0.5 | 0.5231 | 0.831 | 0.527 | 1.0 | 71/143/23 |
| **β=0.6** | 0.5482 | 0.831 | 0.578 | 1.0 | 74/152/11 |
| β=0.8 | 0.5665 | 0.831 | 0.620 | 1.0 | 76/148/13 |
| β=1.0 | 0.5362 | 0.810 | 0.578 | 1.0 | 71/137/29 |

**Honest read:** convex fusion at β=0.6–0.8 beats RRF strongly (MRR +0.060 / +0.096–0.114), improves
Hit@10, and improves far more sessions than it worsens. β=1 (CE-only) drops off → the *combination*
beats either signal. β=0 (raw-satisfaction re-sort) < `sat` (pop-blended order) < RRF, confirming CE
rank-fusion already helps and magnitudes help further.

### Public guardrail (official evaluator, 200 sessions)

| config | HR@10 | MRR | MTTC | TechScore | Δ vs RRF |
|---|---|---|---|---|---|
| RRF (baseline) | 0.960 | 0.838 | 3.015 | 0.8911 | — |
| convex β=0.5 | 0.955 | 0.830 | 3.270 | 0.8810 | −0.010 |
| convex β=0.6 | 0.960 | 0.830 | 3.235 | 0.8843 | −0.0068 |
| convex β=0.8 | 0.955 | 0.818 | 3.270 | 0.8776 | −0.0135 |

> Measurement note: the RRF row here (0.8911) is below the committed `results.json` (0.9298). That
> file predates uncommitted working-tree changes and is treated as stale; the **within-run RRF-vs-
> convex deltas** are the valid guardrail signal (all configs share one agent; DCP is score-neutral in
> the default `INFO_GAIN_MODE="display"` config — personalization is skipped under the satisfaction
> ranker and guidance can't change the `"other"` ask, so the persistent-store pollution from
> experiment runs shifts all rows equally and cancels in the delta).

### Failure analysis
- **Where convex wins (honest):** paraphrased turns where satisfaction ties several lookalikes and the
  cross-encoder's magnitude cleanly separates the true target — median honest rank 2→1, Hit@1 up
  0.628→0.725 (language) / 0.456→0.620 (pillar_free).
- **Where convex loses (public):** leaky turns where the verbatim satisfaction score already pins the
  target at rank 1; the MS-MARCO CE (out-of-domain for e-commerce attributes) injects noise and
  demotes it. This is why every honest-winning β regresses public.
- **Root tension:** the two distributions want opposite β. A single global β cannot satisfy both.

### Latency / resource
CE scoring dominates (~180–220 pairs/s CPU, already paid in both strategies); convex fusion adds
~0.7 ms/turn total across the whole set — negligible. No new model, in-memory, $0.

### Acceptance check
- Honest MRR ≥ +0.01: **PASS** (+0.060 / +0.096 at β=0.6). Honest Hit@10 non-decreasing: **PASS**.
  Median honest rank improves: **PASS**. Latency: **PASS**.
- Public TechScore regression ≤ 0.005: **FAIL** (β=0.6 = −0.0068; no honest-winning β passes).

### Decision — PROMISING — ITERATE
The mechanism is validated on the honest distribution but violates the pre-registered public reject
criterion (>0.005). Per the workflow, the global change is **not shipped**: `USE_CE_CONVEX` defaults
**False** (RRF remains the production fusion); `convex_fuse` + flag + tests are retained.

**Next iteration:** a **gated convex** — apply the CE-magnitude blend only on *uninformative /
paraphrase* turns (reuse the coverage discrimination signal `(top_cov − p90)/top_cov`), so leaky
verbatim turns keep RRF (protect public) and paraphrase turns get the CE precision (capture the honest
win). The forthcoming shadow-private evaluation suite will provide a better proxy than public for the
final β/gate choice.

### Artifacts
`scripts/exp_ce_fusion.py`, `src/retrieval.py::convex_fuse`, `tests/…::ConvexFuseTest`,
`config.py::USE_CE_CONVEX/CE_BETA`.
