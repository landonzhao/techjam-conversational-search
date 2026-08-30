# Judge's Audit Plan — technical competency & comprehensiveness

Written from the seat of the hackathon judge assessing *Technical Execution* and *Comprehensiveness*.
The goal is not a single score but a **per-component verdict**: for each capability the challenge
asks for, is it present, does it work, and is that claim backed by a measurement? A component is
"weak" if it exists but underperforms; "missing" if a required capability is absent; "unproven" if it
exists but nothing measures it.

## The judge's lens

Two questions per component:
1. **Competency** — does it work, and is there a number proving it (not a claim)?
2. **Comprehensiveness** — is the full capability covered, including the hard cases (paraphrase,
   long tail, override, boundary), or only the easy leaky ones?

The evaluator leaks constraints verbatim, so a high public score is necessary but **not sufficient**
evidence of competency. A competent solution must also hold up when the words are reworded. Every
verdict below therefore reads the number **across the leak spectrum**, not just on public.

## Instruments (what isolates each component)

| Instrument | Isolates |
|---|---|
| `eval_default.py` per-scenario metrics | pillar-level strength (Buying/Browsing/Override/Boundary) |
| Leak spectrum: public → `pillar_moderate` → `pillar_free`/leak-free | robustness / real understanding vs leak-exploitation |
| `oracle_leakfree.py` | splits failure into **retrieval** vs **ranking** fault |
| `eval_matrix.py` popularity-ablated column | ranks on **relevance** vs **fame** |
| MTTC per scenario | clarification + routing efficiency |
| Token metering (`usage`) | feasibility / cost |

## Component-by-component audit

For each: what "good" looks like, the diagnostic, and the verdict field to fill from results.

### 1. Intent routing (Buying vs Browsing) — *required*
- Good: correct route per scenario; Buying converges fast, Browsing triggers clarification.
- Diagnostic: per-scenario MTTC and hit on the leak spectrum; does Browsing MTTC stay bounded?
- Fill: Buying vs Browsing MTTC gap; degradation from public → leak-free.

### 2. Conversation state / structured constraints / intent override — *required*
- Good: constraints accumulate; an override on turn 3–4 **drops** the stale constraint.
- Diagnostic: `intent_override` pillar hit/MRR *after* the override turn, across the leak spectrum.
- Fill: override hit vs buying hit; does old constraint contaminate ranking?

### 3. Retrieval (keyword / dense / hybrid) — *required*
- Good: target enters the pool regardless of wording.
- Diagnostic: oracle recall on leak-free.
- Evidence in hand: **recall 99.2%** on leak-free → **STRONG, proven**. Retrieval is not the weakness.

### 4. Ranking / semantic reranking — *required*
- Good: target ranked near the top, not buried by popularity, on reworded input.
- Diagnostic: oracle ranking-fault share; pop-ablated leak-free; MRR across the spectrum.
- Evidence in hand: **97% of honest-set misses are ranking-fault** (target at median pool rank 2);
  popularity buries long-tail targets (leak-free 0.13→0.77 pop-ablated). → **WEAK, the primary gap**,
  with a redesign (satisfaction ranker) validated but not yet defaulted.

### 5. Clarification / question-value estimation — *required + innovation*
- Good: asks the discriminating question for the category; converges in few turns; low MTTC.
- Diagnostic: Browsing pillar hit + MTTC on leak-free; does it ask the slot that unlocks the target?
- Evidence in hand: **Browsing is the weakest pillar on honest data (hit 0.325, MTTC 8.47)**; root
  cause found (cannot form a `feature` question); a fix is built but **measures inert** → **WEAK**.

### 6. Context / personalization (safe profile use) — *required*
- Good: durable preferences bias results when the turn is vague.
- Diagnostic: returning-user set (not yet built); currently dormant in the official eval.
- Fill: **UNPROVEN / dormant** — no measurement exercises it; comprehensiveness gap.

### 7. Orchestration / failure detection / strategy switching — *innovation*
- Good: picks the pipeline per turn; detects a failing strategy and switches.
- Diagnostic: does behavior differ measurably by scenario/phase? Is there a failure-detection signal?
- Fill: present (belief-driven reveal, phase policy) but **failure-detection/strategy-switch is thin**
  — assess whether it is real or decorative.

### 8. Efficiency (latency, token cost) — *feasibility*
- Good: low MTTC, low/zero token cost, deterministic core.
- Evidence in hand: core is $0/offline; token metered; public MTTC 2.87. → **STRONG on cost**;
  honest-set MTTC (7.3) is high because clarification is weak (ties back to #5).

### 9. Explanations (transparent recommendations) — *innovation*
- Good: a rationale for the top pick.
- Fill: `RationaleBuilder` exists — assess whether it is used and meaningful, or vestigial.

### 10. Evaluation / reproducibility — *deliverable*
- Good: a stranger can install, run, and reproduce the number; honest measurement beyond the leak.
- Evidence in hand: measurement harness (leak spectrum, oracle, pop-ablation) is **a genuine
  strength** and rare among entries; reproducibility (README end-to-end) **not yet verified**.

## Scorecard to fill from the results

| Component | Competency (proven?) | Comprehensive (hard cases?) | Verdict |
|---|---|---|---|
| Intent routing | | | |
| State / override | | | |
| Retrieval | proven (recall 99%) | yes | STRONG |
| Ranking | measured | no (buries long tail) | WEAK — primary gap |
| Clarification | measured | no (feature blind) | WEAK |
| Personalization | no | no | UNPROVEN |
| Orchestration | partial | ? | assess |
| Efficiency/cost | proven | — | STRONG |
| Explanations | ? | — | assess |
| Eval/reproducibility | strong harness | README unverified | STRONG (partial) |

## Results (filled) — leak spectrum, current default

| Test set | Verbatim leak | TechnicalScore | hit@10 | MRR | MTTC |
|---|---|---|---|---|---|
| public (official) | ~99% | 0.9172 | 0.970 | 0.899 | 2.87 |
| synthetic | ~99% | 0.8590 | 0.935 | 0.797 | 3.38 |
| pillar_moderate | ~21% | **0.4825** | 0.588 | 0.328 | 6.48 |
| leak-free (language_stress) | ~1% | 0.3825 | 0.444 | 0.290 | 7.33 |
| pillar_free (browsing-heavy) | ~1% | 0.2953 | 0.367 | 0.192 | 8.28 |

**Realistic private-set estimate:** if the organizer adds any meaningful paraphrasing,
`pillar_moderate` ≈ **0.48** is the honest number — nearly half the public score evaporates at ~21%
rewording. Full paraphrase → ~0.30.

**Forensic attribution of the 0.9172 → 0.2953 loss** (score = 0.5·hit + 0.3·MRR + 0.2·eff):
- Ranking (hit + MRR): **−0.51 of −0.62 (~82%)** — recall is 99%, so this is target-retrieved-then-
  ranked-away.
- Clarification (efficiency via MTTC 2.87→8.28): **−0.11 (~18%)**, and it also drags hit down.

## Final verdict

| Component | Verdict | Evidence |
|---|---|---|
| Retrieval | STRONG | recall 99.2% on leak-free |
| Efficiency / cost | STRONG (leaky) | $0 core, MTTC 2.87 public; collapses to 8.28 honest |
| Measurement discipline | STRONG / differentiator | leak spectrum + oracle + pop-ablation |
| **Ranking** | **WEAK — primary gap** | ~82% of the honest-set collapse; fix off by default |
| **Clarification** | **WEAK** | browsing worst pillar 0.325; MTTC 8.28; fix inert |
| **Personalization** | **UNPROVEN** | no measurement exercises it |
| Intent / state-override | adequate | override 0.567 honest > buying 0.475 |
| Orchestration / failure-switch | thin | reveal/phase exist; no real strategy-switch |
| Explanations | assess | RationaleBuilder exists; not shown judge-visible |

**Competency:** high on retrieval/efficiency/measurement; the decision core (ranking + clarification)
only performs on leaked input and more than halves on realistic paraphrase. Diagnosed precisely, but
fixes not yet landed (both behind off-by-default flags; clarification fix still inert).

**Comprehensiveness:** all required pillars present, but ranking and clarification are weak on hard
cases and personalization is unproven — comprehensive in coverage, incomplete in depth.

**Fix priority:** (1) land the satisfaction ranker (validate → default on); (2) debug the clarification
feature-facet probe; (3) build a returning-user set so personalization is at least proven.
