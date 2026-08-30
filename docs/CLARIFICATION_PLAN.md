# Clarification pillar — plan

## Why this pillar (measured)

On the honest (leak-free) set, browsing is our weakest pillar by a wide margin:

```
                 hit@10   mrr    mttc
browsing         0.325    0.177  8.47   ← worst; MTTC near the 10-turn max
buying           0.475    0.364  6.69
intent_override  0.567    0.301  7.10
boundary         0.550    —      6.90
```

Browsing is the pillar that depends on **clarification**: the shopper starts vague and the agent must
ask questions that unlock the discriminators. On the leaky public set browsing is fine (0.988) because
constraints are verbatim; on honest data it collapses.

## Root cause (in the code)

`BeliefModel.update` builds per-attribute uncertainty only over `REQUIRED_SLOTS[category]`
(`src/understanding.py`), and `QuestionSelector.select` asks the highest-uncertainty slot. Two gaps:

1. **No `feature` facet.** `attr_value` extracts only material/color/style/size/use_case/budget. But
   reworded constraints mostly classify as `feature` (evaluator `classify_constraint` falls through to
   `feature`). The agent has no way to form a `feature` question, so it never unlocks those
   constraints — the customer only reveals a constraint when `ask_attribute == classify(constraint)`.
2. **Sparse category map.** `REQUIRED_SLOTS` covers ~16 categories; the long tail (240 categories in
   the test set) falls back to a generic default, so obscure items get generic questions.

## The fix

**Increment 1 (this change): pool-derived feature-facet probe.** When the top candidates disagree on
a distinctive token (a facet not captured by the structured slots), treat that as an askable
`feature` with high uncertainty, and phrase the question around the facet. Category-adaptive by
construction — the facet emerges from the actual pool (jewelry heads surface stone/metal words,
footwear heads surface waterproof/closure words), so it needs no hand-maintained map and covers the
tail. Behind `USE_ADAPTIVE_CLARIFY` (off by default); measured on `pillar_free` browsing + public.

**Increment 2 (next): broaden `REQUIRED_SLOTS` coverage** or derive required slots from the pool's
attribute variance, so the tail categories get sensible structured questions too.

**Increment 3 (later): info-gain ordering** — ask the facet that maximally splits the candidate pool
first, to minimize MTTC.

## Measurement

Primary: `pillar_free.jsonl` browsing slice (MTTC + hit). Guardrail: public (MTTC must not regress).
Diagnostic: `pillar_moderate.jsonl`. All via the official evaluator's per-scenario metrics.

## Debug conclusion — the probe is a dead end (this diagnosis was wrong)

Tracing browsing sessions (`scripts/debug_browsing.py`) overturned the plan:

- `INFO_GAIN_MODE="display"` (the default) makes `next_ask` ignore the selector's `ask_attribute`
  entirely — it only changes message wording. And `ASK_PRIORITY` starts with `"other"`, so the agent
  asks `"other"` every turn regardless, which unlocks any 2 undisclosed constraints per turn.
- So clarification EXTRACTION already works: **constraints are disclosed in 87% of browsing sessions.**
- The browsing failure is RANKING: in **40% of sessions the target is disclosed but never ranked into
  the top 10** (only 7% are a reveal-timing issue).

So attribute-selection (the feature-facet probe) cannot help — the bottleneck is ranking, not asking.
The probe is left behind its off-by-default flag but is inert in display mode; it should be removed in
a cleanup pass. **The real fix is precision reranking of the candidate set — the cross-encoder** (see
ARCHITECTURE §8, `USE_CROSS_ENCODER`), which lifts the honest set pillar_free 0.46 → 0.66 (MRR
0.31 → 0.59) by resolving the exact item among look-alikes once constraints are known.
