# Synthetic-Set Investigation

Public `0.930` → Synthetic (n=1000) `0.742`. The held-out set is broader (shoes, jewelry,
kids, accessories) and includes a hard low-popularity tier. This doc records what we already
know, then digs into the synthetic failures to decide what to actually change.

---

## Part 1 — Gaps already established (before this investigation)

Carried forward from prior analysis this session so we don't re-derive or forget them:

**Information / evaluator limits (not fixable by us)**
- G-A. Public rank-2 losses are largely unwinnable: the discriminative facet (model name,
  pattern, silhouette) is never disclosed, or the two listings are literal duplicates
  (e.g. Memorose 2775 vs 2773 reviews). Info does not exist in the input.
- G-B. `public_0020`-type misses: rating≈1 + only generic constraints disclosed → no signal.

**Retrieval / representation**
- G-C. No fine-grained facet representation (silhouette, neckline, pattern, sleeve, occasion).
- G-D. No visual signal (images) — the primary differentiator for fashion is absent.
- G-E. Synonym table is tiny: 16 hand-written `EXPANSIONS`, 19 `USE_CASE_LEXICON` entries.
  Zero coverage for kids/baby ("toddler", "2T", "newborn"), fashion sub-styles ("bodycon",
  "midi", "peplum", "cocktail"), and occasion→garment inference ("communion", "recital").
- G-F. High-density fashion categories (shirts, tops) showed a real recall hole
  (recall@200 ≈ 0.67–0.70 on the 200-sample probe) — target never enters the pool.

**Ranking**
- G-G. Coverage saturates when many products satisfy the same generic constraints; popularity
  was tie-break-only. Partially fixed this session via `COVERAGE_POP_BLEND=0.1` (big robustness
  win). Residual: near-duplicate ties.
- G-H. No learning-to-rank (correctly avoided — only ~1 label/session; would overfit).
- G-I. LLM reranker implementation is weak: absolute 0–100 scoring on thin input
  (title+material+2 snippets), run over the whole pool. Currently off/neutral.

**Dialogue**
- G-J. Question policy is category-blind: same slot-priority list for a wedding dress and
  running socks. Never asks the category-defining question (age/size for kids, occasion for
  dresses, activity for shoes).
- G-K. `_comparison_phrase` exists but `ask_attribute` stays `"other"`, so discriminative
  elicitation never actually drives the simulator.

**Policy / product**
- G-L. Reveal policy already near its frontier (turn-cap 4 measured optimal); efficiency not
  freely recoverable, coupled to MRR.
- G-M. Diversity measured to cost the scored metric; production/demo-only.

**Housekeeping**
- G-N. `prompts/query_rewrite.txt` is dead (imported nowhere).

---

## Part 2 — Deeper synthetic investigation (measured)

### Finding 1 — Recall is NOT the problem. Reranking is.
On a 300-session stratified sample:
- `recall@200 = 98.7%` (296/300 targets enter the pool; only 4 never retrieved, all "hard"
  tops/shirts). The earlier 0.67 shirt figure was small-sample noise.
- Of the sessions where the target was retrieved but still missed top-10 (53 sessions),
  **62% had the target in retrieval's OWN top-10**, then lost it downstream:

```
Where the target sits in the 200-pool when it MISSES top-10:
  depth 00-09 : 62%   <- retrieval already ranked it top-10; reranking pushed it out
  depth 10-24 : 23%
  depth 25-49 :  8%
  depth 50+   :  8%
```

So the damage is done AFTER retrieval, by the Personalizer -> Coverage -> pop_blend stack.
Retrieval (BM25 + dense + expansion) is doing its job. **Fixing retrieval/synonyms is not the
priority; fixing reranking is.** (Revises G-F: recall is essentially fine.)

### Finding 2 — Coverage reranker demotes sparsely-described (hard-tier) targets.
Coverage scores a product by how many disclosed constraint tokens appear in its text. Hard-tier
targets are sparsely described, so they contain fewer tokens and sink below richly-described
lookalikes — even when retrieval ranked them well. Measured, coverage ON vs OFF by tier:

```
              easy           medium          hard
ON   hit/mrr  0.94/0.73      0.84/0.69       0.76/0.58
OFF  hit/mrr  0.91/0.74      0.94/0.52       0.83/0.36
```

Coverage is a double-edged sword on medium/hard: it pushes some targets OUT of top-10 (hit
drops) but sharpens the rank of the ones it keeps (MRR soars). Net still positive (keep it),
but it **leaks hit@10 on sparse products by fully re-sorting on lexical coverage and overriding
a good retrieval placement.** This is the biggest fixable structural weakness found.
=> Fix direction: bounded demotion — blend the retrieval/dense signal into the coverage sort so
coverage sharpens ranking without sinking a strongly-retrieved sparse target below rank 10.

### Finding 3 — pop_blend is mistuned for low-popularity targets (minor).
`COVERAGE_POP_BLEND` helped the public set (popular targets) but slightly hurts the hard tier
(low-popularity by design): hard-tier MRR 0.61 (blend 0.0) -> 0.58 (blend 0.1). Overall effect
is small (0.05 is marginally best on synthetic). => Consider making the blend
popularity-adaptive, or lower the default; low priority vs Finding 2.

### Finding 4 — MTTC (4.45) is driven by misses, not slow convergence.
First-hit-turn: 29@t1, 76@t2, 55@t3, 74@t4 (reveal cap), then a long tail and 57 misses
(counted as turn 11). The turn-4 cluster is the reveal cap; the misses dominate the average.
=> Rescuing the Finding-2 in-pool misses improves hit AND mttc simultaneously. No separate
efficiency work needed — it falls out of the ranking fix.

### Finding 5 — Boundary scenario is weakest (hit 0.77, mrr 0.46, small n).
Waving off an attribute is handled by adding it to `boundary_attrs` but this does not help
ranking. Low sample count; note but do not over-invest yet.

---

## Part 2b — Fix 1 result (bounded-demotion coverage), measured

Sweeping `COVERAGE_RETRIEVAL_WEIGHT` (w_ret) on the 300-session synthetic sample, by tier:

```
w_ret  SCORE   hit    mrr   mttc  | easy hit/mrr | medium hit/mrr | hard hit/mrr
0.0    0.7621  0.850  0.670 4.19  | 0.94/0.73    | 0.84/0.69      | 0.76/0.58   (current)
0.3    0.8172  0.887  0.757 3.66  | 0.94/0.83    | 0.91/0.76      | 0.80/0.67
0.6    0.8531  0.930  0.784 3.35  | 0.95/0.86    | 0.96/0.78      | 0.88/0.70
1.0    0.8575  0.930  0.796 3.32  | 0.95/0.88    | 0.96/0.79      | 0.88/0.71
```

Confirmed: fusing retrieval order into the coverage sort rescues the well-retrieved targets
coverage was burying. **+0.096 synthetic score**, every tier improves, hard-tier hit 0.76→0.88
and MRR 0.58→0.71, and MTTC drops 4.19→3.32 (efficiency rises for free — Finding 4). w_ret=1.0
best, 0.6 nearly identical. Guardrail (public/robustness) checked before flipping the default.

## Part 3 — Revised priorities (evidence-based)

1. **Bounded-demotion coverage (Finding 2) — highest ROI.** Blend retrieval/dense rank into the
   coverage sort so coverage never sinks a well-retrieved sparse target out of top-10. Directly
   attacks the medium/hard hit loss (the −0.166 hit gap vs public). Measure by tier on synthetic,
   and confirm no regression on public + robustness.
2. **Re-test semantic coverage on the hard tier specifically.** It hurt public (popular targets)
   but sparse items are exactly where lexical coverage fails and embedding similarity may rescue
   recall-into-top-10. Gated, measured by tier.
3. **pop_blend adaptivity (Finding 3) — low effort, small win.** Reduce/soften the blend for
   low-popularity candidates.
4. **Cross-encoder near-tie probe (from prior plan)** — still valid; now measurable on the
   synthetic ranking gap, where it can actually be validated.
5. Retrieval/synonym expansion (kids sizing, fashion sub-styles) — DEPRIORITISED: recall is
   already 98.7%. Only revisit for the 4 true pool-misses if time permits.
6. Category-aware questioning (G-J) — real, but affects MTTC/discovery not the core hit/MRR gap;
   secondary.

Everything measured by difficulty tier on synthetic AND on public + robustness before defaults
flip. The synthetic set is now the primary optimisation target because it exposes the real
weaknesses; public is the guardrail against regression.
