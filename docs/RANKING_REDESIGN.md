# Ranking Redesign — from lexical overlap to need satisfaction

## 0. Problem statement

Our current ranking is, reduced to its essence:

```
rank ≈ verbatim_token_overlap(disclosed_words, product_text) + log(popularity)
```

Two primitive signals — **word presence** and **fame** — applied as successive re-sorts
(`Personalizer` → `CoverageReranker` → pop blend). This is a retrieval-era heuristic, not a
preference model. It has three structural failures:

1. **Verbosity bias.** Coverage counts token *presence*, so it rewards long keyword-stuffed
   listings and penalizes sparsely-described (hard-tier) targets, which contain fewer words —
   even when retrieval placed them top-10 (62% of hard-tier misses are demoted *after* retrieval).
2. **Popularity as relevance.** `POP_WEIGHT` is the "single largest ranking win" — a tell that we
   recommend popular things and score when the target happens to be popular. Wrong on the hard tier.
3. **Structure discarded.** The `NeedModel` has hard/soft/negative constraints with weights and
   polarity; ranking ignores all of it and matches raw tokens. No constraint satisfaction, no
   trade-offs, no profile, no use of the belief model.

Measured consequence: on the leak-free set the score collapses 0.93 → 0.13, because the whole thing
depends on the evaluator leaking verbatim text. **This is an architecture problem, not a weight.**

**Go-ahead evidence — the oracle pass (`scripts/oracle_leakfree.py`, 250 leak-free sessions).** The
headroom is now measured, not assumed:

- **Retrieval recall = 99.2%** — search finds the target on reworded language essentially always
  (including the long-tail items this set over-weights). Understanding is *not* the bottleneck.
- End-to-end hit@10 with our *best* current ranking = 73.6%.
- **Of the 66 misses, 97% (64) are ranking's fault** — the target was in the pool and got ranked
  away; only 3% (2) were never retrieved. On those ranking-fault misses the target's median pool
  rank was **2** — sitting near the top of search, and the word-counting reranker pushed it out of
  top-10.

So the ceiling is **~99% hit@10 vs 73.6% today = 25+ points of pure, reachable ranking headroom**,
living entirely in the stage this redesign replaces. (The deterministic retrieval floor already
banks part of it: leak-free 0.13 → 0.39 → 0.61. This redesign is how we reach for the rest.)

Goal: replace the re-sort stack with a single, inspectable **need-satisfaction scoring function**
that ranks by *how well a candidate satisfies the structured, weighted need*, with semantic matching
and profile priors — and measure it on the leak-free set with popularity ablated.

---

## 1. Design overview

One scorer replaces the `Personalizer → Coverage → pop_blend` stack. For each candidate it returns a
scalar plus a breakdown (for tracing/debuggability):

```
score(cand) =  w_sat  · satisfaction(cand, need)          # constraint satisfaction (primary)
             + w_sem  · dense_relevance(cand)              # semantic retrieval relevance
             + w_prof · profile_prior(cand, profile)       # personalization prior
             + w_pop  · popularity_prior(cand)             # weak fame prior (cold-start only)
```

The weights are **adaptive to need specificity** (see §4), which is what demotes popularity from
"primary signal" to "cold-start prior." `satisfaction` is the new heart of the system.

Layering (unchanged direction): retrieval produces the 200-pool *with dense scores retained*; the
scorer consumes pool + `NeedModel` + profile + dense scores and emits the final order. No stage can
clobber another because every signal is a *term*, not an overriding re-sort.

---

## 2. The satisfaction function (F1 + F2 — the core)

For each constraint `c = (slot, value, polarity, weight)` in the `NeedModel`, compute a graded
per-attribute match of the candidate against `c`:

```
match(c, cand) ∈ [0, 1]   — how well cand satisfies the attribute
state(c, cand) ∈ {SATISFIED, CONFLICT, UNKNOWN}
```

**Extract the candidate's value for the slot**, tiered by what we can determine:

- **Structured slots** (`material`, `color`, `size`, `budget`, `category`): reuse the existing
  extractors (`attr_value` in `understanding.py`, `MATERIAL_RE`, `COLOR_RE`, price field). Compare
  the candidate's extracted value to `c.value`:
  - exact normalized match → `match=1.0`, `SATISFIED`
  - synonym / expansion-table / embedding-neighbor match → `match≈0.8`, `SATISFIED`
  - candidate has a *different, conflicting* value → `match=0.0`, `CONFLICT`
  - candidate metadata is silent → `match=0.5`, `UNKNOWN`  ← **key: absence ≠ mismatch**
- **Free-form slots** (`feature`, `style`, `use_case`): graded match, not token presence:
  - `lexical = IDF-weighted(present value-tokens) / IDF-weighted(all value-tokens)`
  - `semantic = cosine(embed(c.value), product_embedding)`  (product embeddings already cached)
  - `match = max(lexical, α·semantic)`; `SATISFIED` if above a threshold, else `UNKNOWN`
    (free-form rarely yields a hard `CONFLICT`).

**The `UNKNOWN` vs `CONFLICT` distinction is the fix for verbosity bias.** Coverage treats a missing
word as 0 (punishing sparse targets). We treat missing metadata as *neutral* (0.5) and only punish an
*actual conflict*. A sparse target with thin text is no longer demoted for being terse — only for
genuinely mismatching.

**Polarity:**
- want (`+1`): contribution `= +signed`, where `signed = match`
- avoid (`−1`): contribution `= −match` when the candidate HAS the attribute; `0` when absent.

**Importance weighting** (per constraint):

```
importance(c) = c.weight · idf_boost(c.value)
    c.weight    : hard 1.0 / soft 0.5 (already in NeedModel; ContextDistiller recency-decays it)
    idf_boost   : rare attribute values matter more than generic ones ("2T" >> "black")
```

**Aggregate (normalized — the fix for verbosity bias at the sum level):**

```
satisfaction = Σ_c importance(c)·signed(c, cand)  /  Σ_c importance(c)     ∈ [−1, 1]
```

Dividing by total importance makes scores comparable across candidates and sessions, so a verbose
product cannot win by accumulating more matched tokens — only by *satisfying more of what matters*.

**Hard-constraint gate:** if any constraint with `c.weight ≥ HARD_THRESHOLD` is in state `CONFLICT`,
apply a strong multiplicative demotion (near-disqualify). A `UNKNOWN` hard constraint does *not* gate
(we don't punish missing metadata). Gate only fires on confident conflicts (conservative, because our
structured extractors are imperfect).

---

## 3. The other terms

- **`dense_relevance(cand)`** — the candidate's normalized dense-retrieval score (already computed;
  retain it through retrieval instead of discarding it). This is the paraphrase-robust backbone:
  it survives when verbatim matching fails, and being a *term* it can no longer be clobbered by
  coverage (fixes gap #7 and the leak-free collapse partly, deterministically).
  **This term is empirically the highest-leverage move we have found.** Fusing the retrieval order
  back as a floor (`COVERAGE_RETRIEVAL_WEIGHT`, the shippable precursor to this term) lifts the
  leak-free set **0.125 → 0.385 (w=1) → 0.605 (w=2)** while public slides only 0.9297 → 0.9171 →
  0.8923 — public MRR even *rises* at w=1. By contrast, routing paraphrase through **LLM slot
  extraction scored net-negative on the same set (0.1251 → 0.1211) at 5× latency**, because an
  extracted slot dead-ends into the lexical matcher while dense retrieval is the real semantic
  mechanism. The dense term generalizes the floor: retrieval relevance as an additive signal that is
  never thrown away. (Numbers: `scripts/exp_retrieval_weight.py`, LLM-understanding experiment.)
- **`profile_prior(cand, profile)`** (F4) — see §5.
- **`popularity_prior(cand)`** — `log1p(rating_number)`, normalized. A prior, weighted by `w_pop`
  which is *high only when the need is underspecified* (§4).

---

## 4. Adaptive weights (F3 — demote popularity)

Let specificity `s = clip(Σ importance(discriminating constraints) / S_ref, 0, 1)` — how much
distinctive signal the user has disclosed. Then:

```
w_sat  = 0.3 + 0.6·s        # lean on satisfaction as the need sharpens
w_sem  = 0.4 − 0.2·s        # semantic backbone matters most when vague
w_prof = 0.3 − 0.1·s        # profile fills the gap when vague, steps aside when specific
w_pop  = 0.3·(1 − s)        # fame is a COLD-START prior only; → 0 as the user discloses
```

(Numbers are starting points to sweep on the leak-free set, not tuned constants.) This directly
fixes the hard-tier bias: an unpopular but specifically-described target is ranked on satisfaction,
not fame; popularity only helps on turn 1 when we know almost nothing.

---

## 5. Profiles as a real preference prior (short + long term)

Today the only profile→rank path is `TAG_WEIGHT·keyword_overlap` — a rounding error. Replace it:

- **Short-term (within session):** already handled by `ContextDistiller`, which recency-decays
  constraint weights in the `NeedModel`. Because `importance(c)` uses `c.weight`, a repeatedly
  emphasized attribute automatically gains ranking weight and a stale one fades. Wire the decayed
  weights straight into `importance` (they already exist; we just stop ignoring them).
- **Long-term (across sessions):** `UserProfile.prefs` (durable `(slot, value, weight)` with time
  decay) and `category_affinity` become the `profile_prior`:

  ```
  profile_prior(cand) = Σ_p p.weight · match(p, cand)  /  Σ_p p.weight
                        + β · category_affinity[cand.category]
  ```

  i.e. "does this candidate match the user's durable tendencies (materials, colors, price band,
  categories they gravitate to)." Dominant when the current turn is vague (`w_prof` high), negligible
  when the user is specific.

**Honest caveat (must be in the writeup):** long-term profiles are **dormant in the official eval**
because public/private sessions are distinct users, so this term cannot move the leaderboard score.
Its value is (a) product-facing realism and (b) robustness on returning users. Therefore it is
measured on a **synthetic returning-user set** (same user across 2–3 sessions), not the public set,
and ships default-low so it never harms the scored path.

---

## 6. Belief-aware hedging (F5 — phase 2, optional)

The `BeliefModel` computes per-attribute uncertainty and is currently consumed only by dialogue.
Feed it to ranking: when uncertainty for slot X is high, *down-weight X's contribution* (avoid
over-committing to one value) and optionally hedge the top-k so a plausible target isn't locked out
by a guess. Keep this a later refinement — it is a polish on top of F1–F4, not a prerequisite.

---

## 7. Implementation plan

New module `src/ranking.py → NeedSatisfactionScorer` (sibling of `CoverageReranker`, which stays
available behind a flag during transition). Gated by `USE_SATISFACTION_RANKER` (default False until
it beats the guardrails).

- **Phase 0 — measurement first (blocking). ✅ DONE.** The scoreboard exists:
  - `scripts/eval_matrix.py` — {config} × {normal, popularity-ablated} over leak-free / public /
    synthetic in one command.
  - `scripts/oracle_leakfree.py` — the ranking oracle; established the 25-point headroom above.
  - Remaining nice-to-have: a per-attribute satisfaction report (which slots satisfied on hit vs
    miss) — build alongside Phase 1 for debuggability, not a blocker.
**Empirical design correction (probe on 3 honest sessions).** The regex-parsed `state.need`
constraints are unreliable on paraphrase — they hallucinated `size=m` on every session, emitted
`brand=other`/`feature=have`, and missed the reworded attributes entirely. The real signal lives in
the raw `state.constraint_phrases` ("made of a rich napped pile", "indoor strength sessions"). So the
scorer's PRIMARY input is the raw phrases, matched by `max(verbatim-lexical, semantic-cosine)` — which
also makes it a strict generalization of `CoverageReranker` (coverage = the verbatim-lexical term
alone). Structured `need` constraints are used only for the reliable, low-noise signals: negatives
(avoid) and budget. This is Phase 1 as-built.

- **Phase 1 — satisfaction core (THE NEXT STEP, decisive).** Add `NeedSatisfactionScorer.match()`
  with structured + graded matching, UNKNOWN/CONFLICT states, normalized aggregate, hard gate.
  Score = `w_sat·satisfaction` only (no other terms yet), so the comparison is clean. **Race it
  head-to-head against `CoverageReranker` on `eval_matrix.py`, popularity-ablated.** Concrete steps:
  1. `NeedSatisfactionScorer` class in `src/ranking.py`; consume `state.need` constraints + cached
     product embeddings; return `(ordered_asins, per_candidate_breakdown)`.
  2. `match(constraint, cand)` reusing `understanding.py` extractors for structured slots
     (`material`/`color`/`size`/`budget`) and IDF-lexical + embedding-cosine for free-form slots.
  3. Flag `USE_SATISFACTION_RANKER` in `config.py`/`agent.py` (default False); wire as an alternate
     to the coverage call in `agent.respond`.
  4. Add the config as a row in `eval_matrix.py`; run pop-ablated. **Decision gate:** keep only if it
     beats coverage on leak-free pop-ablated while public holds ≥ ~0.92. If it loses, stop and keep
     the retrieval floor — the oracle says the headroom is real, so a clean loss means our `match()`
     is wrong, not that the ceiling is missing.
- **Phase 2 — full scorer.** Add `dense_relevance`, `popularity_prior`, adaptive weights (§4).
  Retain dense scores through retrieval so the scorer can consume them.
- **Phase 3 — profile prior.** Long-term `profile_prior`; build the returning-user synthetic set;
  measure warm-start lift there.
- **Phase 4 — belief hedging** (optional, §6).

Each phase ships behind the flag, measured, kept only on a clean guardrail (public ≥ ~0.92) plus a
leak-free/synthetic-hard gain. `ARCHITECTURE.md` updated in the same change.

---

## 8. Data flow (after Phase 2)

```
respond()
  ├─ retrieval  → pool[200], dense_score[asin]   (dense scores RETAINED, not discarded)
  ├─ need       = NeedModel (regex + optional LLM slots), recency-decayed by ContextDistiller
  ├─ profile    = UserProfile (durable prefs, category affinity)   [long-term, default-low]
  └─ NeedSatisfactionScorer.rank(pool, need, profile, dense_score, specificity)
        for each cand: score = w_sat·satisfaction + w_sem·dense + w_prof·profile + w_pop·pop
        return ordered pool + per-candidate breakdown (→ tracer)
```

`Personalizer` (popularity, tag overlap) is absorbed as scorer terms and retired as a separate stage.
`CoverageReranker` becomes the `SATISFIED`-exact-match tier inside `match()` (coverage is the special
case of satisfaction where matching is verbatim), so leaky-set behavior is preserved by construction.

---

## 9. Why this fixes each gap

| Gap | Fix |
|---|---|
| Verbosity bias (coverage counts presence) | UNKNOWN≠CONFLICT; normalized aggregate |
| Popularity as relevance | adaptive `w_pop → 0` as need sharpens |
| Structure discarded | per-constraint match with polarity + hard/soft weight + hard gate |
| No trade-off model | weighted normalized satisfaction over the full constraint set |
| Profiles don't reach ranking | `profile_prior` term (long-term) + decayed weights (short-term) |
| Belief unused in ranking | uncertainty down-weighting (phase 4) |
| Stack of clobbering re-sorts | single additive scoring function; retrieval relevance is a term |
| Optimized for the leak | leak-free set is primary; pop-ablated measurement |

---

## 10. Risks & guardrails

- **Must not tank public.** The leak rewards verbatim coverage; satisfaction's exact-match tier
  reproduces it, so on leaky data behavior is preserved. Verified by the public guardrail each phase.
- **Imperfect structured extraction** could misfire the hard gate. Mitigate: gate only on confident
  CONFLICT, never on UNKNOWN; keep the gate conservative and ablatable.
- **Semantic term cost.** Product embeddings are cached; embedding a few constraint values per turn
  is cheap. No new heavy dependency; core stays deterministic and offline ($0).
- **Profile term is dormant in eval** — ship default-low, measure on the returning-user set only.
- **Scope creep.** Phases are independently shippable; F1 alone (satisfaction vs coverage on the
  leak-free set) is the decisive first experiment. Do not build phases 2–4 until F1 measures a gain.

---

## 11. Definition of done

- A single `NeedSatisfactionScorer` produces the ranking, with an inspectable per-candidate breakdown.
- Leak-free score materially above the 0.13 baseline **with popularity ablated** (proving relevance,
  not fame), while public holds ≥ ~0.92.
- Every ranking claim backed by a leak-free + pop-ablated + per-attribute number in `docs/`.
- `ARCHITECTURE.md` describes the scorer as the ranking stage; coverage documented as its exact tier.
