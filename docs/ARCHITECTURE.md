# Shopping Copilot — Architecture

A conversational shopping agent for the TechJam 2026 "AI Conversational Search and
Recommendations" challenge. It holds a multi-turn conversation with a shopper and, over a
maximum of 10 turns, surfaces the right product from a frozen ~50,000-item Amazon
Clothing/Shoes/Jewelry catalog — either recommending products or asking a clarifying question.

This document explains how the system works and where to change things. For how to run it, see
the [README](../README.md). For the ranking redesign in progress, see
[RANKING_REDESIGN.md](RANKING_REDESIGN.md).

---

## 1. System overview

The organizer's evaluator drives the agent through a single entry point:

```python
Agent.respond(session_id: str, user_message: str, turn: int, top_k: int) -> dict
# returns {"message": str, "ask_attribute": str|None, "recommendations": [asin, ...], "usage": {...}}
```

Each call is one conversation turn. Internally a turn flows through five stages:

1. **Understand** the message → update the session's structured need.
2. **Route** intent (buying vs browsing).
3. **Retrieve** a candidate pool (~200) from the catalog.
4. **Rank** the pool so the best match is on top.
5. **Decide** whether to ask a clarifying question or reveal recommendations, then compose the reply.

The core system is **deterministic, offline, and needs no API key** — dense retrieval uses
precomputed local embeddings. An optional LLM layer exists (Gemini) but is **off/inert on the
scored path** by default; see §10.

### A note that explains the whole design: the evaluator leaks

The official evaluator builds each shopper brief by copying constraints **verbatim** from the
target product's own catalog text (~99.7% of constraint tokens appear in the target's listing). A
ranker can therefore win by matching words, without understanding anything. Our public score
(~0.93) partly reflects this leak.

Because a real shopper (and possibly the private test set) paraphrases, we built a **leak-free
held-out set** (`data/language_stress_set.jsonl`) that rewords every attribute into vocabulary that
does *not* appear in the target's text. Much of the ranking code, and our measurement discipline,
exists to be robust on that honest distribution — not just the leaky one. Keep this split in mind:
**public = leaky guardrail; leak-free = honest generalization.**

---

## 2. High-level pipeline

```
Agent.respond(session_id, message, turn, top_k)
        │
        ▼
  Session state  ── ConversationState (per session_id)
        │
        ▼
  Understanding ── extract_constraints() → constraint_phrases (raw text)
        │          SlotFiller.parse()    → NeedModel constraints (structured slots)
        │          [optional LLM slot extraction — off by default]
        ▼
  Intent router ── IntentRouter.score() → buying vs browsing
        │
        ▼
  Retrieval ────── BM25 (catalog.py) ⊕ Dense (VectorRetriever/BGE) ⊕ expansion
        │          fused via RRF → ~200 candidates                (recall@200 ≈ 99%)
        ▼
  Ranking ──────── Personalizer (popularity + profile pre-sort)
        │          then ONE of:
        │            • CoverageReranker.rerank_scored   (default)
        │            • NeedSatisfactionScorer.rank       (flag, the rebuild)
        │          then apply_negatives() filter
        ▼
  Belief ───────── BeliefModel.update() → confidence, per-attribute uncertainty
        │
        ▼
  Dialogue policy ─ ask a clarifying question  ── OR ──  reveal recommendations
        │           (QuestionSelector)                    (adaptive reveal count)
        ▼
  Response ─────── compose_message() [deterministic; optional LLM phrasing]
        │          + token-usage delta
        ▼
     returned dict
```

LLM touchpoints (all optional, off/inert by default): slot extraction, use-case inference,
response phrasing, an LLM reranker. None is on the scored path in the default configuration.

---

## 3. Repository structure

```
src/
  agent.py           Orchestrator. respond() runs the whole pipeline. Owns the FLAG LEDGER.
  catalog.py         Catalog loading; text()/terms() helpers; BM25 (SQLite FTS5 index).
  dialogue.py        ConversationState, IntentRouter, extract_constraints(), compose_message(),
                     phase_transition(), next_ask().
  understanding.py   NeedModel + Constraint, SlotFiller, QuestionSelector, BeliefModel,
                     attr_value(), converge(), apply_negatives(), EXPANSIONS/USE_CASE_LEXICON,
                     CatalogVocab.
  retrieval.py       VectorRetriever (dense BGE embeddings), rrf() fusion, vector_weight().
  ranking.py         Personalizer, CoverageReranker, NeedSatisfactionScorer, Diversifier.
  context_engine.py  ContextDistiller (short-term), ProfileService/UserProfile (long-term),
                     OrchestrationPolicy, GuidanceLearner.
  llm_inference.py   OPTIONAL: LLMSlotExtractor, SmartUseCaseInferencer, LLMResponseGenerator.
  reranker.py        OPTIONAL: CrossEncoderReranker, LLMReranker.
  keys.py            GeminiClientPool — key rotation + process-wide token metering.
  config.py          All tunable constants (weights, thresholds, flags, model ids).
  trace.py           Structured per-turn tracer (debugging).

evaluator/           Official local evaluator + robustness paraphraser (a hard contract — read-only).
data/                catalog.jsonl, public_set.jsonl, synthetic_set.jsonl, language_stress_set.jsonl.
scripts/             Experiments + measurement harness (§11).
tests/               Unit tests (pytest).
docs/                This file, RANKING_REDESIGN.md, STRENGTHENING_PLAN.md, findings.
```

Dependency direction: `agent` → domain services (`understanding`, `retrieval`, `ranking`,
`context_engine`, `dialogue`) → infrastructure (`catalog`, `keys`, `config`). No circular imports;
`config` is a leaf.

---

## 4. Request lifecycle (one turn)

Walking `Agent.respond` (`src/agent.py`):

1. **Session lookup.** `self._sessions[session_id]` → `ConversationState`. `reset()` starts a new
   session (clears turns, need, belief; loads any user profile).
2. **Constraint capture.** `extract_constraints(message)` appends raw phrases to
   `state.constraint_phrases`. `SlotFiller.parse(message)` produces structured `Constraint`s merged
   into `state.need` (a `NeedModel`) via `revise()`. *Optional* LLM slot extraction can add more,
   but is off by default and inert on the scored path (§10).
3. **Intent routing.** `IntentRouter.score()` yields a buying-vs-browsing score, smoothed across
   turns (`CONFIDENCE_EMA`) into `state.intent`.
4. **Context distillation** (optional). `ContextDistiller.update()` recency-decays constraint
   emphasis; `OrchestrationPolicy.plan()` picks per-turn behavior.
5. **Retrieval.** `_retrieve()` builds a pool of ~`POOL_SIZE` candidates by fusing BM25 and dense
   results (§7).
6. **Personalization pre-sort.** `Personalizer.rerank()` nudges by popularity + profile tags
   (skipped when the satisfaction ranker owns ordering).
7. **Ranking.** Either `CoverageReranker.rerank_scored()` (default) or
   `NeedSatisfactionScorer.rank()` (flag) reorders the pool (§8). `apply_negatives()` then filters
   out candidates that carry an avoided attribute.
8. **Belief update.** `BeliefModel.update()` turns the ranked pool + scores into confidence and
   per-attribute uncertainty.
9. **Dialogue decision.** If the belief is uncertain and a question would help, `QuestionSelector`
   picks an `ask_attribute`; otherwise the agent reveals.
10. **Adaptive reveal.** `_reveal_count()` decides *how many* items to return: a short list (often
    1) while still unsure, the full `top_k` once confident, constraints stop arriving, or the turn
    cap is reached (`REVEAL_TURN_CAP`). This exploits the evaluator's first-appearance MRR rule
    (§11) and applies identically to the private set.
11. **Diversify** (browsing only). `Diversifier.reorder()` (MMR) spreads the tail so it isn't ten
    near-identical items.
12. **Compose + meter.** `compose_message()` builds the reply text (deterministic; optional LLM
    phrasing). Token usage since turn start is attached as `usage`.

---

## 5. Conversation state

`ConversationState` (`src/dialogue.py`) is the per-session memory. Key fields:

- `all_text` — the turn-by-turn message history.
- `constraint_phrases` — **raw** disclosed phrases (e.g. "made of a rich napped pile"). This is the
  primary ranking signal — it survives paraphrase, unlike the parsed slots.
- `need` — a `NeedModel`: a list of structured `Constraint(slot, value, polarity, weight, turn)`
  plus the resolved `category`.
- `belief` — a `Belief`: confidence + per-attribute uncertainty (drives clarification).
- `intent`, `buying_score`, `phase`, `ctx`, `profile` — routing, phase, distilled context, profile.

**Constraint model.** `slot ∈ {material, color, size, style, budget, use_case, category, feature,
brand}`; `polarity` +1 want / −1 avoid; `weight` 1.0 hard / 0.5 soft. `NeedModel.revise()` is a
non-monotonic merge: a new value on the same `(slot, value)` supersedes the old (so "actually, not
down" flips a prior "down"), while different values on one slot coexist (multi-value slots).
Intent-override turns let the newest requirement dominate.

**Honest caveat.** On paraphrased input the regex `SlotFiller` is unreliable (it may hallucinate a
default size or miss a reworded material). That is *why* ranking leans on `constraint_phrases` +
semantic matching rather than the parsed slots — see §8.

---

## 6. Intent routing

`IntentRouter` (`src/dialogue.py`) scores a message on a buying↔browsing axis from lexical/task
signals and the breadth of the query. The score is smoothed across turns
(`buying_score = α·raw + (1−α)·prev`) so a single ambiguous message doesn't flip intent. Intent
tunes downstream behavior: retrieval weighting (browsing leans dense/semantic, buying leans BM25),
personalization strength, and diversification. Override scenarios set `intent = "override"` and let
the newest constraint take over.

---

## 7. Retrieval

Goal: **recall** — get the target into the ~200 pool. It succeeds ~99% of the time, including on
paraphrase (measured: `recall@200 ≈ 99.2%` on the leak-free set), so retrieval is *not* the
bottleneck — ranking is (§8, §11).

Two retrievers, fused:

- **BM25** (`src/catalog.py`, SQLite FTS5) — lexical keyword match over catalog text.
- **Dense** (`VectorRetriever`, `src/retrieval.py`) — cosine similarity over precomputed
  **BGE-small-en-v1.5** 384-d embeddings (cached in `cache/`, built by
  `scripts/build_embeddings.py`). This is the paraphrase-tolerant retriever: it maps "keeps the rain
  out" near "waterproof" by meaning. `search_decayed()` recency-weights multi-turn queries.

Fusion is **Reciprocal Rank Fusion** (`rrf()`); the dense/BM25 mix is set by `vector_weight()` from
the buying score. An optional low-weight query-expansion track (synonym `EXPANSIONS`) adds recall
for reworded queries. The pool size is `POOL_SIZE` (phase-adjusted).

Per-turn cost note: the catalog embeddings are precomputed once; only the *query* is embedded live
each turn (the main CPU cost of an eval run).

---

## 8. Ranking

Retrieval maximizes recall; ranking maximizes ordering quality (Hit@10, MRR). This is where the
target — usually already near the top of retrieval — must be kept on top. All ranking lives in
`src/ranking.py`.

### 8.1 Personalizer
A light pre-sort blending log-popularity and profile-tag overlap into the retrieval order. **Caveat
learned by measurement:** a flat popularity pre-sort is the dominant reason correct *long-tail*
targets get buried on paraphrased turns (leak-free score collapses when popularity is on). It helps
the leaky public set but hurts the honest set, so it is treated as a lever, not a default good.

### 8.2 CoverageReranker (default ranker)
Scores each candidate by **verbatim coverage** of the raw `constraint_phrases` — the IDF-weighted
fraction of disclosed phrase tokens present in the candidate's catalog text, with a full-phrase
bonus. On the leaky distribution this singles out the exact ASIN among near-duplicates and drives
the ~0.93 public score. It supports several measured, configurable refinements:

- **Retrieval floor** (`COVERAGE_RETRIEVAL_WEIGHT`, **on by default = 1.0**). When coverage matches
  nothing (paraphrase), it would otherwise collapse to a popularity tie-break and *discard* the good
  semantic order retrieval produced. The floor RRF-fuses the retrieval order back in as a safety
  net. Measured: lifts leak-free 0.125 → 0.385 for only −0.013 public. **This is our proven, shipped
  robustness win.**
- **Discrimination gate** (`COVERAGE_INFORMATIVE_MIN`, **off by default = 0**). Applies the floor
  *only* when coverage failed to single out the target — measured by whether the top candidate
  stands out from its pool rivals (`top − p90`), not by raw magnitude (which a shared brand anchor
  fools). Experimental: lifts leak-free further but costs public.
- **Paraphrase pop-suppression** (`SUPPRESS_POP_ON_PARAPHRASE`, off): zero popularity on
  uninformative turns so the order falls back to retrieval, not fame.
- **Structured coverage, price proximity, phrase tiers** — additional optional signals (off/low by
  default); see `config.py`.

The returned score dict is always *raw* coverage, so the belief model sees true constraint coverage.

### 8.3 NeedSatisfactionScorer (the rebuild — flag `USE_SATISFACTION_RANKER`, ON by default)
A generalization of coverage that scores each candidate by how well it *satisfies* the disclosed
phrases:

```
match(phrase, candidate) = max( verbatim_lexical(phrase, candidate),      # IDF fraction present
                                SEM_ALPHA · semantic_cosine(phrase, candidate) )
satisfaction(candidate)  = phrase-length-weighted mean of match over phrases
```

Coverage is exactly the special case that keeps only the lexical term; adding the semantic term
(cosine to the candidate's cached embedding, via `VectorRetriever.phrase_similarity_matrix`) is what
survives paraphrase. No popularity re-sort — a well-satisfied but unpopular target is not demoted.
**Phase 2** adds *adaptive popularity* (`SATISFACTION_POP_WEIGHT`): a fame prior weighted
`(1 − specificity)`, high when the shopper is vague, fading to ~0 as they get specific.

Validated (`scripts/validate_satisfaction.py`, `SATISFACTION_POP_WEIGHT=0.15`) and now the **default
ranker**: it lifts the honest sets (pillar_free 0.295 → 0.398 = +35%, pillar_moderate 0.483 → 0.501)
while public holds at 0.903 — a deliberate −0.014 leaderboard cost for large paraphrase robustness.
Set `USE_SATISFACTION_RANKER=False` to revert to pure coverage (public 0.9172). See RANKING_REDESIGN.md.

### 8.4 Diversifier
Maximal-marginal-relevance reordering of the tail (browsing), protecting a confident head so the
likely target is untouched. Trades relevance vs novelty (`DIVERSITY_LAMBDA`).

---

## 9. Clarification

Clarification is a *decision*, not just text. `BeliefModel` computes per-attribute uncertainty;
`QuestionSelector` (`src/understanding.py`) chooses the attribute whose answer would most reduce
ambiguity, and `converge()`/`next_ask()` decide *whether* asking beats revealing now. Because MTTC
(mean turns to conversion) is scored, the policy avoids unnecessary questions: it reveals as soon as
confidence is high enough or the turn budget is short. `ask_attribute` in the response tells the
evaluator what was asked.

---

## 10. Context, personalization, and the LLM layer

**Short-term context** (`ContextDistiller`, `src/context_engine.py`): recency-decays constraint
emphasis within a session and prunes stale signals, so a repeatedly-stressed attribute carries more
weight and an abandoned one fades.

**Long-term profile** (`ProfileService`/`UserProfile`): durable per-user preferences and category
affinity with time decay. **Honest caveat:** the official eval uses distinct public/private users,
so long-term profiles are dormant there and ship default-low; their value is realism and
returning-user robustness, not the leaderboard.

**LLM layer** (`src/llm_inference.py`, `src/reranker.py`, all on `GeminiClientPool`): slot
extraction, use-case inference, response phrasing, and an LLM reranker. **All optional and off/inert
on the scored path by default.** Measured finding: on the leak-free set (where paraphrase is the
whole task) LLM slot extraction scored *net-negative* at 5× latency — because an extracted slot
dead-ends into a lexical matcher, whereas dense retrieval already handles paraphrase for free. So
the LLM is not tuned for leaderboard points. Every Gemini call is metered (`usage_metadata`) into
process-wide counters and reported per turn as `response["usage"]`, so the evaluator sees exact
token cost (cached calls cost zero).

---

## 11. Evaluation

Run the official evaluator (see README) to get `recommended_technical_score`:

```
TechnicalScore = 0.50·HitRate@10 + 0.30·MRR + 0.20·Efficiency
Efficiency     = clip((11 − MTTC) / 10, 0, 1)
```

Which components move which metric:

| Metric | Primarily driven by |
|---|---|
| Hit Rate@10 | retrieval recall (§7) + ranking (§8) |
| MRR | ranking / reranking (§8) — the redesign targets this |
| MTTC → Efficiency | intent routing + clarification policy + adaptive reveal (§6, §9, §4) |

**Measurement harness (important).** Ad-hoc scores on the leaky public set flatter us. Two scripts
give honest numbers:

- `scripts/eval_matrix.py` — {ranking config} × {normal, popularity-ablated} over leak-free /
  public. The pop-ablated column exposes whether a config ranks on *relevance* or *fame*.
- `scripts/oracle_leakfree.py` — the ranking oracle. It established that on the honest set retrieval
  recall is 99.2%, end-to-end hit@10 is ~74%, and **97% of misses are ranking's fault** (target was
  in the pool, ranked away, median pool rank 2). That ~25-point gap is the reachable ranking
  headroom the redesign pursues.

Other sets: `data/synthetic_set.jsonl` (harder leaky diagnostic), `data/language_stress_set.jsonl`
(the leak-free honest set, built by `scripts/build_language_stress_set.py`).

---

## 12. Configuration

All tunables live in `src/config.py` (weights, thresholds, model ids, flags). Runtime feature flags
are class attributes on `Agent` (the "FLAG LEDGER" docstring marks each CORE vs OPTIONAL). Nothing
tunable is hardcoded in algorithms. Secrets (`GEMINI_API_KEY*`) come only from the environment
(`.env`, never committed); the core runs with no key.

Notable defaults (see comments in `config.py` for measured tradeoffs):

| Constant | Default | Meaning |
|---|---|---|
| `COVERAGE_RETRIEVAL_WEIGHT` | 1.0 | retrieval floor (proven robustness win) |
| `COVERAGE_INFORMATIVE_MIN` | 0.0 | discrimination gate (off; experimental) |
| `USE_SATISFACTION_RANKER` | True | the ranker rebuild (default; validated) |
| `SATISFACTION_SEM_ALPHA` | 1.0 | semantic-vs-lexical weight in satisfaction |
| `SATISFACTION_POP_WEIGHT` | 0.15 | adaptive popularity prior (Phase 2) |
| `POOL_SIZE` | — | retrieval pool size |

---

## 13. Extension guide

| To change… | Work in… |
|---|---|
| Buying vs browsing detection | `IntentRouter` in `src/dialogue.py` |
| BM25 behavior | `src/catalog.py` (FTS5 index) |
| Dense retrieval / embeddings | `src/retrieval.py`; rebuild via `scripts/build_embeddings.py` |
| Retrieval fusion weights | `vector_weight()` in `src/retrieval.py`; `config.py` |
| Conversation memory / slots | `NeedModel`/`SlotFiller` in `src/understanding.py` |
| Constraint capture (raw phrases) | `extract_constraints()` in `src/dialogue.py` |
| Coverage ranking / retrieval floor / gate | `CoverageReranker` in `src/ranking.py` |
| The ranker rebuild (satisfaction) | `NeedSatisfactionScorer` in `src/ranking.py` |
| Clarification strategy | `QuestionSelector`/`converge()` in `src/understanding.py` |
| Adaptive reveal (MRR/MTTC) | `_reveal_count()` in `src/agent.py` |
| Personalization / profiles | `src/context_engine.py` |
| LLM slot/use-case/response/rerank | `src/llm_inference.py`, `src/reranker.py` |
| Thresholds / weights / flags | `src/config.py` + the FLAG LEDGER in `src/agent.py` |
| Measuring honestly | `scripts/eval_matrix.py`, `scripts/oracle_leakfree.py` |

---

## 14. Key design decisions

- **Retrieval and ranking are separate.** Recall is ~99%; the problem is ordering. Keeping them
  distinct let us prove (via the oracle) that ranking, not understanding, is the bottleneck.
- **Rank on raw phrases + meaning, not parsed slots.** The regex slot filler is noisy on paraphrase;
  the raw `constraint_phrases` matched by `max(lexical, semantic)` is the reliable signal — and makes
  the satisfaction ranker a clean superset of coverage.
- **Popularity is a lever, not a default good.** It helps the leaky set but buries long-tail targets
  on the honest set; hence the retrieval floor, pop-suppression, and adaptive-popularity work.
- **Robustness ships behind flags, measured.** The proven flat retrieval floor is on; newer
  mechanisms (gate, satisfaction, adaptive pop) are off by default with documented tradeoffs, so the
  known-good public score is never silently degraded.
- **The LLM is optional and honest.** It is not on the scored path because it measured net-negative
  there; it stays available for language robustness, fully token-metered.
- **Determinism and reproducibility.** Core is offline, seedable, and needs no API key.

---

## 15. Known limitations

- **Public score reflects the evaluator leak.** ~0.93 overstates real-world performance; the honest
  (leak-free) number is far lower, and the ranking redesign targets that gap.
- **Regex slot extraction is brittle on paraphrase** (hallucinated defaults, missed rewordings). The
  ranking path routes around it, but the parsed `NeedModel` itself is not trustworthy on reworded
  input.
- **Long-term personalization is unexercised by the official eval** (distinct users).
- **The satisfaction ranker is not yet the default** — validated as a balanced win but pending final
  guardrail confirmation before it ships on.

---

## 16. Future work

- Finish the ranking redesign (RANKING_REDESIGN.md): validate and default the satisfaction ranker;
  add structured 3-state matching (SATISFIED/CONFLICT/UNKNOWN) and a hard-constraint gate; tune
  adaptive popularity.
- Category-adaptive clarification (ask the discriminating question for the category) to cut MTTC on
  the long tail.
- Wire belief uncertainty into ranking (hedge under ambiguity).
- A returning-user synthetic set to actually exercise long-term profiles.
```
