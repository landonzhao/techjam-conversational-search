# Strengthening Plan — Shopping Copilot

Master roadmap. Sequences the ranking redesign (`RANKING_REDESIGN.md`) and the innovation
work (`INNOVATION_AND_EXECUTION_PLAN.md`) into one prioritized plan, grounded in what we have
now *measured* rather than what we hoped. Every workstream states its change, its measurement,
and its exit criterion. Nothing ships without a leak-free number and a public guardrail.

---

## 0. Where we actually stand (evidence, not vibes)

Established by experiment this cycle:

1. **The graded distribution leaks.** Public and (almost certainly) private sessions materialize
   constraints verbatim from the target's own spec sheet (99.7% leak). Our `language_stress_set`
   (leak-free, held-out paraphrase, one fair anchor per session) is the **only honest generalization
   measure we have**. Public = leaderboard guardrail; leak-free = robustness scoreboard.
2. **Ranking, not retrieval, is the bottleneck.** Recall@200 = 98.7%. 62% of hard-tier misses are
   *in retrieval's own top-10 and then demoted by re-sorting*. The target is usually already there;
   we rank it away.
3. **Dense retrieval already is our paraphrase understanding.** It maps reworded language to the
   right neighborhood semantically, for free, at retrieval time.
4. **Preserving that semantic order is a large, free, deterministic win.** `COVERAGE_RETRIEVAL_WEIGHT`
   sweep (leak-free / public):
   `0.0 → 0.125/0.9297` · `0.5 → 0.251/0.9216` · `1.0 → 0.385/0.9171` · `2.0 → 0.605/0.8923`.
   **Wired `w_ret = 1.0`** (`config.py:109`): leak-free tripled, public −0.013, public MRR *rose*.
5. **LLM understanding on the scored path is settled: it isn't a win.** On the leak-free set (the one
   place understanding should matter most) full LLM slot extraction scored **0.1211 vs 0.1251
   baseline — net-negative — at 5× latency** (1996s). It stays optional, off by default. We stop
   trying to make it earn leaderboard points.

**One-line thesis:** our real leverage is *ranking that uses semantic and structured signal as
first-class terms*, not more understanding bolted onto a lexical matcher. `w_ret` is the first proof;
the `NeedSatisfactionScorer` generalizes it.

---

## P0 — Lock the free win, build the scoreboard (this week, blocking)

### P0.1  Make the retrieval floor *conditional* (the near-free version of `w_ret`)
`w_ret = 1.0` costs −0.013 public because it applies the floor even when verbatim coverage is
*correctly* winning (leaky case). We can likely keep the leak-free gain **and** recover public by
applying the floor **only when coverage is uninformative**:

- Detect a paraphrased / no-verbatim-signal turn: top coverage score is near-zero
  (`max_coverage < ε` → nothing matched verbatim across the pool).
- When coverage *has* signal (verbatim matches present, the leaky case) → floor OFF (protect public).
- When coverage is flat (paraphrase, the private-risk case) → strong floor (lean on dense order).

**Key leverage from the audit:** because the gate turns the floor *off* on verbatim sessions, the
paraphrase branch can use a *higher* weight than the flat 1.0 (which had to stay low to protect
public). So conditional is not "recover the −0.013" — it targets **best-of-both-columns**: keep
public ≈ 0.9297 *and* reach the `w_ret=2.0` leak-free level (≈0.60), because the two branches never
apply to the same session. The per-candidate coverage scores (`exact` in `rerank_scored`) are already
computed, so the gate statistic (`max(exact.values())`) is free.

Change: `src/ranking.py` (`rerank_scored`) gates `retrieval_weight` on the coverage-informativeness
statistic instead of applying it as a constant; new config `COVERAGE_INFORMATIVE_MIN` (the ε
threshold). **Exit:** public ≥ 0.928 **and** leak-free ≥ 0.50 (strictly dominating flat `w_ret=1.0`,
which gave 0.9171/0.385). If it can't beat flat on both, keep the constant. Do this first.

### P0.2  Measurement harness (Phase 0 of `RANKING_REDESIGN.md`) — blocking for all ranking work
Before touching the scorer, build the instruments so every later claim is falsifiable:
- **Primary**: leak-free set. **Guardrail**: public. **Diagnostic**: synthetic-hard.
- **Popularity-ablated mode** (`w_pop = 0`): exposes whether we rank on *relevance* or *fame*.
- **Per-attribute satisfaction report**: which slots were satisfied on hit vs miss sessions.
- **Ranking oracle**: best achievable rank in the pool given the target's true attributes →
  quantifies headroom (how much of the miss is rank-able-away vs genuinely absent).

Deliver as `scripts/eval_matrix.py` (one command → the full table). **Exit:** one reproducible
command prints public / leak-free / synth-hard × {normal, pop-ablated} plus the oracle ceiling.

### P0.3  Record the settled findings in the docs (same change)
- `ARCHITECTURE.md` LLM-status section: state plainly that LLM slot extraction measured net-negative
  on paraphrase and dense retrieval is the paraphrase mechanism — so nobody re-litigates it.
- `RANKING_REDESIGN.md`: fold the `w_ret` and LLM numbers in as the empirical motivation for the
  `dense_relevance` term.

---

## P1 — The decisive ranking experiment (satisfaction vs coverage)

This is the one experiment that validates or kills the whole redesign. Do **only** Phase 1 first.

- Build `NeedSatisfactionScorer` (`src/ranking.py`, sibling of `CoverageReranker`, flag
  `USE_SATISFACTION_RANKER`, default off).
- Implement `match(c, cand)` with the **UNKNOWN ≠ CONFLICT** state machine and the
  **importance-normalized aggregate** (`RANKING_REDESIGN.md §2`). Score = `w_sat·satisfaction`
  *only* — no other terms yet, so the comparison is clean.
- Reuse existing extractors (`understanding.py` `attr_value`, `MATERIAL_RE`, `COLOR_RE`, price field)
  for structured slots; IDF-lexical + cached-embedding cosine for free-form slots.
- **Head-to-head vs coverage** on all three sets, pop-ablated included.

**Exit:** satisfaction beats verbatim coverage on leak-free **with popularity ablated** (proves
relevance, not fame), while public holds ≥ ~0.92 (coverage's exact-match tier lives *inside*
`match()`, so leaky behavior is preserved by construction). **If it doesn't beat coverage here, stop
— we've spent two days, not two weeks, and we keep `w_ret` as the ranking win.**

---

## P2 — Full scorer (only on a measured P1 gain)

Replace the re-sort stack with the single additive function
`score = w_sat·satisfaction + w_sem·dense + w_prof·profile + w_pop·pop`:

- **`w_sem·dense`**: retain dense scores through retrieval instead of discarding them; this is
  `w_ret` generalized into a term that *cannot be clobbered* (fixes the "re-sort stack" gap).
- **Adaptive weights** (`RANKING_REDESIGN.md §4`): `w_pop = 0.3·(1−s)` with specificity `s` — demotes
  popularity from primary signal to cold-start prior. Directly fixes the hard-tier fame bias.
- Retire `Personalizer` and the pop-blend as separate stages (absorbed as terms).

**Exit:** leak-free strictly above the P1 number, public ≥ ~0.92, and an inspectable per-candidate
score breakdown wired to the tracer (debuggability is a deliverable, not a nicety).

---

## P3 — Profiles that actually reach ranking (honest about eval dormancy)

- **Short-term**: `ContextDistiller` already recency-decays constraint weights; wire the decayed
  weights straight into `importance(c)` — an emphasized attribute gains ranking pull, a stale one
  fades. Nearly free; measure on multi-turn sessions.
- **Long-term**: `UserProfile.prefs` + `category_affinity` → the `profile_prior` term, dominant when
  the turn is vague, silent when specific.
- **Honesty requirement**: long-term profiles are **dormant in the official eval** (distinct
  users) — ship default-low, and measure warm-start lift on a purpose-built **returning-user
  synthetic set** (same user, 2–3 sessions). Value = realism + judged "personalization" pillar,
  not the leaderboard. Say so in the writeup.

---

## P4 — Innovation & dialogue pillars (parallelizable, from `INNOVATION_AND_EXECUTION_PLAN.md`)

Independent of the ranking track; picks up the judged non-Technical pillars. Sequence by leverage:
- **Category-adaptive clarification**: ask the discriminating question *for that category* (a ring
  needs metal/stone; a boot needs waterproofing/use) instead of a generic slot list. Lowers MTTC on
  the long tail where we currently ask blind. Work in `src/clarification/`.
- **Info-gain ask-vs-display**: ask only when a question's expected reduction in candidate ambiguity
  beats showing results now — protects MTTC (the Efficiency term) from needless turns.
- **Belief-aware hedging in ranking** (`RANKING_REDESIGN.md §6`, Phase 4): under high attribute
  uncertainty, down-weight that slot so we don't lock the true target out on a guess.

**Exit:** each shows a measured MTTC or leak-free improvement; none regresses public. Do not build
speculative "innovation" that we can't measure.

---

## What we are explicitly NOT doing (and why)

- **LLM on the scored path** — measured net-negative on paraphrase at 5× cost. Optional/off only.
- **More same-kind synthetic sets** — the leak-free set is the honest measure; another leaking set
  just re-confirms the exploit. Only new set worth building is the returning-user one (P3).
- **Semantic coverage / heavier vector infra** — measured to hurt paraphrase robustness before;
  dense retrieval + the retrieval floor already deliver the semantic win deterministically and $0.
- **Chasing the last decimal of public** — public overstates real performance (verbatim leak). We
  protect it as a guardrail (≥ ~0.92) but optimize the leak-free/robustness axis the private set and
  the judges actually reward.

---

## Sequencing summary

| Order | Workstream | Type | Gate to next |
|---|---|---|---|
| 1 | P0.1 conditional `w_ret` | free, deterministic | public ≥ 0.928 & leak-free ≥ 0.33 |
| 2 | P0.2 measurement harness | instrument | one-command eval matrix runs |
| 3 | P0.3 doc the settled findings | honesty | — |
| 4 | **P1 satisfaction vs coverage** | **decisive** | beats coverage pop-ablated, public ≥ 0.92 |
| 5 | P2 full additive scorer | build-out | leak-free > P1, public ≥ 0.92 |
| 6 | P3 profiles into ranking | realism | warm-start lift on returning-user set |
| 7 | P4 clarification / hedging | pillars | measured MTTC or leak-free gain |

P0 is this week and blocking. P1 is the fork in the road: everything after it is conditional on it
measuring a real gain. That discipline is the point — we ship measured wins and stop at the first
one that doesn't pay.
