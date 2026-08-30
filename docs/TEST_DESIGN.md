# Test-Set Design — judging the agent by the pillars

Written from the judge's seat. The organizer scores four scenario types separately (Buying 40%,
Browsing 40%, Intent Override 15%, Boundary 5%) and reports Hit@10 / MRR / MTTC per type. The spec
also says paraphrasing may be added and "cannot decide correctness — hits are exact code matches."
So a good test set must (a) match that scenario mix, (b) isolate each pillar so strength/weakness is
attributable, and (c) **not be winnable by verbatim word-matching** — because the public set is ~99%
verbatim-leaked and flatters any lexical ranker.

Guiding principle: **measure understanding and dialogue, not string overlap.** Every session keeps
one fair identifying anchor (a brand or distinctive noun a real shopper would say) so the target is
findable, but every *attribute* is reworded into held-out vocabulary (verified disjoint from the
catalog text and from our own synonym tables). We also vary the leak level on purpose (a spectrum),
so we can see the degradation curve rather than a single number.

---

## What each pillar tests, and how the session is built

### 1. Buying (40%) — retrieval + ranking under paraphrase
**Claim under test:** given a specific, hard requirement stated in the shopper's own words, can the
agent find and *rank* the exact target?
**Construction:** turn 1 discloses an anchor + one hard, discriminating constraint, reworded
("a rich napped pile" for corduroy). Soft preferences follow. No clarification needed — this isolates
retrieval recall and ranking quality.
**Judge reads:** Hit@10 and especially **MRR** (is the target near the top, not just in the ten?).
**Failure it exposes:** a lexical ranker that collapses to popularity when the reworded words don't
match — the exact failure our oracle found (target retrieved at rank ~2, ranked away).

### 2. Browsing (40%) — clarification policy + convergence (MTTC)
**Claim under test:** starting vague, does the agent ask *useful, discriminating* questions and
converge quickly — rather than generic questions or premature guesses?
**Construction:** turn 1 is deliberately vague ("something nice for summer evenings"). The
discriminating attributes are **withheld** and only revealed when the agent asks for that specific
slot (the evaluator's customer policy answers `ask_attribute`). So an agent that asks the *right*
slot for the category (a boot → waterproofing; a ring → stone) converges in few turns; a
category-blind agent wastes turns.
**Judge reads:** **MTTC / Efficiency** first, then Hit. This is the pillar most tied to clarification
quality, and the one our current agent is weakest on.
**Failure it exposes:** asking low-value or off-category questions → high MTTC.

### 3. Intent Override (15%) — state management / non-monotonic update
**Claim under test:** when the shopper replaces a preference on turn 3–4 ("actually, ignore that — I
want X"), does the agent *drop* the stale constraint and re-rank toward the new one?
**Construction:** establish a reworded preference early; inject the override message on turn 3 or 4
with a new, conflicting constraint. Per spec, the session cannot convert before the override.
**Judge reads:** Hit/MRR *after* the override turn, and whether the old constraint still contaminates
ranking.
**Failure it exposes:** monotonic state that keeps the old constraint → ranks the wrong item.

### 4. Boundary (5%) — graceful "no preference" handling / not over-asking
**Claim under test:** when the shopper genuinely has *no* preference on an attribute, does the agent
handle a "no preference" answer gracefully — not loop on it, not treat the non-answer as a
constraint — and still converge?
**Construction:** mark one requested attribute as `no-preference`; when the agent asks for it, the
customer declines. The target is otherwise identifiable.
**Judge reads:** MTTC (no wasted turns) and Hit. Robustness, not raw accuracy.
**Failure it exposes:** re-asking the declined attribute, or filtering on a non-constraint.

---

## Cross-cutting axes (applied to every scenario)

- **Leak spectrum.** Generate the same targets at three leak levels so we see the curve, not a point:
  - `leaky` — verbatim (mirrors the public set; upper bound / sanity).
  - `moderate` — some attributes reworded, some verbatim (realistic).
  - `leak-free` — all attributes reworded (honest floor, the private-set risk).
- **Long tail.** Sample across distinct coarse categories, over-weighting obscure buckets, since
  head categories dominate a naive per-product sample and hide tail weakness.
- **Personalization (optional).** A returning-user variant repeats a user across sessions with a
  consistent tag (petite / eco-materials / budget) to test whether safe personalization helps —
  dormant in the official eval (distinct users) but a real robustness signal.

---

## How we measure on these sets

The official evaluator runs directly on any set that carries its own `intent_card` + `behavior`
(materialized hidden fields), so the standard metrics and the per-scenario breakdown apply unchanged.
On top of that:

- `scripts/eval_matrix.py` adds the **popularity-ablated** view (relevance vs fame).
- `scripts/oracle_leakfree.py` splits failures into **retrieval vs ranking**.
- The generator reports the **measured leak rate** per set, so "less leak" is a number, not a claim.

The result is a scorecard that reads like a judge's: per-pillar Hit/MRR/MTTC, on an honest
distribution, with failure attributable to a specific stage.
