# Remediation Plan — evidence-based

Supersedes earlier speculative drafts. Every item here traces to a measured finding in
`docs/SYNTHETIC_FINDINGS.md`. Baselines: public 200 `0.930`, synthetic 1000 `0.742`,
robustness (paraphrase) `0.691`.

---

## 1. The failures, ranked by measured impact

| # | Failure | Evidence | Fixable? |
|---|---------|----------|----------|
| F1 | **Coverage reranker demotes well-retrieved, sparsely-described targets out of top-10** | 62% of in-pool misses were in retrieval's own top-10, then pushed out; coverage OFF raises hard-tier hit 0.76→0.83, medium 0.84→0.94 | **Yes — primary** |
| F2 | Lexical coverage fails on sparse items (few tokens to match) | hard-tier is sparsely described by construction; coverage score ∝ token count | Yes |
| F3 | `pop_blend` mistuned for low-popularity targets | hard-tier MRR 0.61→0.58 as blend 0.0→0.1 | Yes (minor) |
| F4 | Near-duplicate / near-tie targets ranked 2–8 | rank-2 forensics on public; medium/hard MRR | Partly |
| F5 | Ambiguous cases where the discriminating detail is never disclosed | Crocs Freesail vs Classic; duplicate listings; `public_0020` | **No** |
| F6 | Boundary scenario weakest (hit 0.77, mrr 0.46) | small n; wave-off doesn't affect ranking | Yes (secondary) |

Recall is **not** a failure (98.7% of targets reach the pool). Retrieval and synonym
coverage are adequate; the damage is in reranking. This corrects the earlier assumption
that we needed better query understanding / synonym tables.

The score gap between public (0.93) and synthetic (0.74) is almost entirely F1+F2: the
public set is popular, well-described, findable clothing — exactly the case where lexical
coverage works — so it hides the weakness that the broad set exposes.

---

## 2. Fix 1 (primary) — bounded-demotion coverage

### The problem, precisely
`CoverageReranker.rerank_scored` fully re-sorts the 200 candidates by lexical coverage:

```python
ranked[a] = coverage[a] + pop_blend * pop(a)
key(a)    = (-ranked[a], -pop(a), base_rank[a])   # retrieval order only a final tie-break
```

Because coverage is the primary key and `base_rank` (the retrieval ranking) is only consulted
on exact ties, a sparsely-described target that retrieval ranked #3 can sink to #50 purely
because its short description contains fewer of the disclosed tokens than richer lookalikes.
We measured this is where hit@10 leaks on medium/hard.

### The design
Coverage's value is real (it roughly doubles hard-tier MRR for items it keeps), so we do not
weaken it — we **bound how far it can demote a strongly-retrieved item.** Retrieval already
had an opinion; coverage should refine it, not overwrite it.

Three candidate mechanisms, to be measured against each other:

**Option A — RRF fusion of coverage-order and retrieval-order (preferred; reuses `rrf()`).**
Rank the pool by coverage, rank it by retrieval (incoming order), and reciprocal-rank-fuse:
```
fused_score(a) = 1/(k + coverage_rank(a)) + w_ret * 1/(k + retrieval_rank(a))
```
A target retrieval loved keeps a floor even if its coverage is weak; a strong-coverage item
still rises. `w_ret` is the single knob (0 = today's behaviour; higher = more retrieval
protection). Clean, bounded, parameter-light.

**Option B — retrieval-score blend into the primary key.**
`ranked[a] = coverage[a] + β · norm_retrieval_score(a)` where `norm_retrieval_score` is a
normalised 1/(base_rank+1). Simpler but mixes two different score scales; needs care.

**Option C — demotion cap.**
An item in retrieval's top-K may be demoted by coverage by at most M positions. Most direct
statement of intent, but a hard rule with two knobs and edge cases; keep as fallback.

Start with **Option A**. It is the smallest, most principled change and reuses existing fusion.

### Where the code changes
- `src/ranking.py` → `CoverageReranker.rerank_scored`: add `retrieval_weight` param; when > 0,
  compute the fused order instead of the pure-coverage sort. Return raw `coverage` scores
  unchanged (the belief model must still see true coverage, as today with `pop_blend`).
- `src/config.py`: `COVERAGE_RETRIEVAL_WEIGHT` (new knob).
- `src/agent.py`: pass the flag through the existing `rerank_scored` call; add
  `USE_BOUNDED_COVERAGE` ablation toggle.

### The tension to resolve by measurement
Too little retrieval protection → no hit recovery on sparse items. Too much → dilutes the
coverage MRR sharpening. The sweep must find the `w_ret` that **recovers medium/hard hit@10
without losing the coverage MRR gain**, and does not regress public.

### Measurement (mandatory, by tier)
Sweep `w_ret ∈ {0, 0.3, 0.6, 1.0}` and report, per difficulty tier on a ≥300 synthetic
sample AND on public 200 AND robustness:
- hit@10 and MRR per tier
- overall TechnicalScore on each set
Keep the setting that maximises synthetic TechnicalScore subject to **public not regressing
below ~0.928** and **robustness not regressing below ~0.685**. Flip default only then.

### Success criterion
Medium/hard hit@10 rises toward the coverage-OFF levels (medium ~0.94, hard ~0.83) while MRR
stays near coverage-ON levels (hard ~0.58). Expected synthetic gain: the largest single move
available, because F1 is the dominant gap.

---

## 3. Fix 2 — semantic coverage for sparse items (targeted re-test)

Lexical coverage fails precisely when the target's text is sparse (F2). Embedding similarity
between the constraint phrases and the product does not depend on token overlap, so it can
rescue sparse targets. We already built `VectorRetriever.phrase_similarities` and the
`semantic_weight` path in `rerank_scored`; it was measured to hurt on the *public* set (popular,
well-described targets — semantic flattened the signal), so it is off.

**Hypothesis:** it helps exactly where lexical fails — the sparse hard/medium tier — and hurts
where lexical already works — easy/public. If so, it should be **gated to low-coverage items**,
not applied globally.

**Plan:** re-test `USE_SEMANTIC_COVERAGE` on synthetic split by tier. If the tier hypothesis
holds, apply semantic coverage only when a candidate's lexical coverage is below a threshold
(i.e. as a rescue signal for sparse items), never as a global term. Measure by tier + guardrails.
Lower priority than Fix 1 and partly overlapping with it (both target sparse-item hit).

---

## 4. Fix 3 — popularity-adaptive coverage blend (low effort)

`COVERAGE_POP_BLEND=0.1` helps popular targets (public) and slightly hurts obscure ones
(hard tier, F3). Make the blend **soft-cap on candidate popularity**: apply the popularity
boost with diminishing weight for already-popular items and near-zero for the tail, so it
breaks ties among plausible matches without burying a low-popularity correct target. One-line
change to the blend term; sweep and measure by tier. Small expected gain; do after Fix 1 since
Fix 1 may absorb part of it.

---

## 5. Fix 4 — cross-encoder / listwise LLM rerank on the near-tie band (F4)

Still valid, now measurable on the synthetic ranking gap (unlike the leaked public set).
- **Probe first (cheap):** enable the already-coded `CrossEncoderReranker` gated to
  `belief.margin < τ`, top-8, and measure on synthetic by tier + public + robustness.
- **Decision gate:** only if the probe lifts the near-tie band do we invest in the stronger
  **listwise** LLM reranker (rewrite the weak absolute-scoring prompt in `src/reranker.py` to
  return a best→worst ordering, fed the differentiating tokens, fired only on the near-tie band,
  with counted token cost and the cross-encoder as offline fallback).

Sequence this **after** Fix 1, because Fix 1 changes which sessions are still near-ties.

---

## 6. What we will NOT do (and why)

- **Rebuild retrieval / expand synonym tables for kids/fashion vocabulary.** Recall is 98.7%;
  this is not where we lose. (Revisit only for the 4 true pool-misses if time remains.)
- **Learning-to-rank.** ~1 label/session; overfits. Hand-tuned weights validated across three
  sets are correct at this data scale.
- **Variant dedup.** Can convert a rank-2 into a miss by dropping the target member.
- **Diversity in the scored list.** Measured to cost hit/MRR; production/demo flag only.
- **Chase F5 (withheld-detail ambiguity).** The information does not exist in the input; effort
  there is wasted on the eval. Belongs in the demo narrative, not the ranker.

---

## 7. Measurement protocol (applies to every fix)

1. Every change ships behind an ablation flag, default unchanged until proven.
2. Report **by difficulty tier** on a ≥300 synthetic sample — aggregate numbers hide the
   tier-specific effects that matter (a change can help easy and hurt hard and look "neutral").
3. Guardrails: public must not regress below ~0.928; robustness not below ~0.685.
4. The synthetic set is now the **primary optimisation target** (it exposes the real weakness);
   public and robustness are regression guardrails.
5. Keep only measured wins. Revert plausible-but-unproven changes (the standard that already
   killed semantic-coverage-global, override-reset, and IDF-coverage this project).

---

## 8. Sequence and expected outcome

| Order | Item | Effort | Primary metric moved | Gate |
|---|---|---|---|---|
| 1 | Fix 1 bounded-demotion coverage (Option A, sweep w_ret) | M | medium/hard hit@10 | keep if synthetic↑, public≥0.928 |
| 2 | Fix 2 semantic coverage gated to sparse items | M | sparse-tier hit | tier hypothesis must hold |
| 3 | Fix 3 popularity-adaptive blend | S | hard-tier MRR | small; after Fix 1 |
| 4 | Fix 4 cross-encoder probe → listwise LLM (if probe passes) | S→M | near-tie MRR | probe is the gate |
| 5 | Fix 6 boundary handling | S | boundary scenario | secondary |

**Expected:** Fix 1 is the big move — it should recover much of the −0.166 hit gap and, because
misses also drive MTTC, improve efficiency for free (Finding 4). Fixes 2–4 are incremental.
Realistic target: close a meaningful share of the public↔synthetic gap (0.74 → high-0.7s/low-0.8s)
without regressing public, proving the system is robust across product types, not just tuned to
the leaked clothing set.

---

## 9. Why this is the right plan

The synthetic set converted guesswork into diagnosis. We are not failing at finding products or
understanding language — we are failing at one specific thing: **a reranker that overrides good
retrieval and buries thinly-described products.** The plan fixes that directly, measures by the
tier that exposes it, guards the sets that must not regress, and refuses the expensive or
unmeasurable detours. It is small, targeted, and evidence-led — not a rewrite.
