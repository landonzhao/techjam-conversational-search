# Shopping Copilot — Final Technical Architecture

## 1. Executive summary

This repository implements a text-only conversational shopping agent for the TechJam Shopping
Copilot challenge. The agent searches a frozen 50,000-product Amazon Clothing, Shoes and Jewelry
catalog and must return the correct `parent_asin` within ten turns. Its central engineering problem
is not ordinary keyword search. It is maintaining the shopper's *current* intent while the shopper
hesitates, contradicts themselves, rejects recommendations, changes product categories, or supplies
catalog-like evidence.

The final design is built around four ideas:

1. **An Ordered Preference Ledger** records every preference mutation as `SET`, `ADD`, `REMOVE`,
   `CLEAR`, or `NO_PREFERENCE`. It preserves an audit trail while exposing only the active state to
   retrieval and ranking.
2. **Negation priority and cascade clearing** make corrections non-monotonic. A later category or
   value can retire an earlier one, and modifiers attached to an abandoned category do not leak into
   the replacement request.
3. **A Dual-Track Query Builder** uses a correction-safe ledger projection for ordinary, reworded
   conversation, but can recover the historical raw-transcript retrieval behavior when runtime
   catalog evidence strongly indicates an exact, catalog-derived disclosure.
4. **A Unified Multi-Stage Ranker** fuses retrieval order, need satisfaction, several carefully
   gated exact-coverage signals, and a conditional popularity prior. The exact path is strong only
   when exact evidence exists; the semantic path otherwise protects the hybrid retrieval head.

These components solve two apparently conflicting objectives:

- **Real conversational correctness:** abandoned terms such as `boot`, `hiking`, `linen`, `red`, and
  `blue` disappear from the effective need and clean search query.
- **Competition compatibility:** exact catalog phrases in the released public simulator remain
  useful enough to satisfy the public score guardrail, without making append-only raw history the
  default behavior for honest, paraphrased conversations.

The system is deterministic and fully offline on its scored path. Dense retrieval and LLM helpers
are optional and degrade gracefully when their model/cache or credentials are unavailable.

---

## 2. Design goals and invariants

The implementation follows these invariants.

### 2.1 Current intent is authoritative

The latest explicit session correction outranks:

- an earlier turn;
- an earlier phrase in the same turn;
- inferred attributes from an abandoned product branch;
- an LLM fallback extraction from the same turn; and
- a durable DCP/profile preference learned in a previous session.

The full ledger remains available for debugging, but inactive entries are never treated as active
positive evidence.

### 2.2 Retrieval and ranking consume the same effective state

Structured constraints, phrase-level evaluator disclosures, the query text, personalization, and
ranking must agree about which values are active. A value cannot be removed from the visible state
while silently surviving in the BM25 query, dense query, coverage phrases, or profile tags.

### 2.3 Missing metadata means unknown, not false

Amazon listings are sparse and inconsistent. A missing `material`, `color`, brand, or description is
not evidence that the product violates a request. Sparse products retain their retrieval evidence and
are demoted only by positive contrary evidence, not by absent fields.

### 2.4 Public compatibility must be earned from runtime evidence

No dataset label is passed into the agent. The leaky/public-compatible path activates only from the
user's text plus exact evidence found in the catalog. Ordinary conversation stays on the clean path.

### 2.5 Evaluation runs are isolated and comparable

Every benchmark run uses a fresh temporary DCP directory, disables persistence, pins the candidate
pool independently from popularity ablations, and uses the official 40/40/15/5 scenario mix.

---

## 3. System overview

The official entry point is `Agent` in [`src/agent.py`](../src/agent.py), exposed through
[`starter/agent.py`](../starter/agent.py). The public interface is fixed:

```python
Agent(catalog_path)
agent.reset(session_id, user_profile)
response = agent.respond(session_id, user_message, turn, top_k)
```

The response contains:

```json
{
  "message": "...",
  "ask_attribute": "color",
  "recommendations": [{"parent_asin": "..."}],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0}
}
```

### 3.1 End-to-end data flow

```text
user turn
   |
   v
turn-1 catalog-leak probe -------------------------------+
   |                                                     |
   v                                                     |
SlotFiller.parse() + optional LLM slots                  |
   |                                                     |
   v                                                     |
NeedModel.revise(): ordered ledger                       |
 SET / ADD / REMOVE / CLEAR / NO_PREFERENCE              |
   |                                                     |
   +--> phrase-history invalidation                      |
   +--> durable-profile masking                          |
   +--> active category + active constraints             |
   |                                                     |
   v                                                     |
ConversationState.query_text()                           |
   |                                                     |
   +--> clean: ledger + active phrases + safe fallback   |
   +--> leaky: raw accumulated history <-----------------+
   |
   v
BM25 + optional BGE dense retrieval + expansion
   |
   v
RRF candidate pool (normally 200)
   |
   v
DualTrackRanker.rank()
 retrieval + satisfaction + gated exact coverage
 + legacy coverage order + cumulative coverage
 + current-turn n-grams + conditional popularity
   |
   v
belief update -> clarification selector -> full Top 10 in evaluator mode
```

### 3.2 Ownership by module

| Concern | Primary implementation |
|---|---|
| Catalog loading, tokenization, BM25 | `src/catalog.py` — `Catalog`, `terms()` |
| Dense retrieval and reciprocal-rank fusion | `src/retrieval.py` — `VectorRetriever`, `rrf()` |
| Slot parsing and preference state | `src/understanding.py` — `SlotFiller`, `NeedModel` |
| Query projection and dialogue state | `src/dialogue.py` — `ConversationState` |
| Unified ranking | `src/ranking.py` — `DualTrackRanker` and supporting scorers |
| Turn orchestration and track selection | `src/agent.py` — `Agent.respond()`, `Agent._retrieve()` |
| DCP session/profile memory | `src/context_engine.py` |
| Ranking/retrieval constants | `src/config.py` |
| Isolated benchmark construction | `scripts/eval_support.py` |
| Current score matrix | `scripts/eval_matrix.py` |
| Retrieval-versus-ranking oracle | `scripts/oracle_leakfree.py` |

---

## 4. Ordered Preference Ledger

### 4.1 Why an append-only need model failed

Natural conversation is not monotonic. Consider:

> “I want a fluffy slipper—wait, no, a bucket hat. Red... actually blue... yellow is better.”

Appending every recognized keyword produces an impossible request containing slippers, hats, red,
blue, and yellow. It also contaminates retrieval because BM25 and dense search continue seeing terms
the shopper explicitly abandoned.

The final model therefore separates:

- **`NeedModel.ledger`** — the immutable-in-order audit history of all preference events; and
- **`NeedModel.constraints`** — the active compatibility view consumed by the rest of the agent.

Both are implemented in [`src/understanding.py`](../src/understanding.py).

### 4.2 Ledger event schema

`Constraint` is the event record:

| Field | Meaning |
|---|---|
| `slot` | `category`, `material`, `color`, `size`, `style`, `budget`, `use_case`, `feature`, or brand |
| `value` | Normalized value, for example `shoe`, `yellow`, or `waterproof` |
| `polarity` | `+1` wanted, `-1` rejected, `0` control operation |
| `weight` | Hard versus soft preference strength |
| `turn` | Source turn number |
| `operation` | One of the five ledger operations |
| `span` / `surface` | Character position and original wording for ordered repair handling and masking |
| `source` / `confidence` | Regex, inference, or LLM provenance |
| `active` | Whether the event still contributes to the current need |

The character span is essential. A single user message may contain several revisions; applying
events in textual order lets the last valid repair win without discarding the audit trail.

### 4.3 Operation semantics

`NeedModel.revise()` is the single update contract for deterministic and optional LLM extraction.

| Operation | State transition |
|---|---|
| `SET(slot, value)` | Replaces active positive values in an exclusive slot. Used for category, brand, color, size, and budget, and after a repair cue. |
| `ADD(slot, value)` | Adds a compatible value. Used for genuinely multi-valued preferences such as features or explicit conjunctions. |
| `REMOVE(slot, value)` | Deactivates the matching positive and records an active negative. |
| `CLEAR(slot)` | Removes all active values for a slot without declaring a boundary preference. `CLEAR(__modifiers__)` clears category-dependent modifiers. |
| `NO_PREFERENCE(slot)` | Clears the slot and records that the shopper does not want to answer it. The question selector must not ask it again. |

Exclusive slots overwrite by default. A genuine conjunction can still be represented: `red and
blue` becomes `SET(color, red)` followed by `ADD(color, blue)`, while `red, actually blue` becomes
two ordered `SET` events with only blue active.

### 4.4 Repair cues and ordered parsing

`SlotFiller.parse()` recognizes discourse markers through `REPAIR_CUE_RE` and
`_REPAIR_BETWEEN_RE`, including:

- `actually` and the common `actly` shorthand;
- `wait`, `wait no`, and `wait nah`;
- `instead` and `rather`;
- `scratch that` and `never mind`;
- `changed my mind`;
- `make that`, `make it`, and `I mean`;
- `nah` and contrastive `but`.

The parser extracts all relevant facts, sorts them by character span, examines the connector between
successive values, and assigns `SET`, `ADD`, or `REMOVE`. A trailing “scratch that” without a
replacement emits a `CLEAR` for the last active slot.

### 4.5 Same-turn negation priority

The self-contradiction bug occurred in:

> “Instead of linen, I want polyester.”

The material regex saw both words and could emit positive `linen` after a negative extraction.
There are now two defenses:

1. `SlotFiller.emit()` calls `_explicitly_negated()` before ordinary local polarity logic. Patterns
   such as `instead of X`, `rather than X`, `no X`, `not X`, `without X`, `avoid X`, and
   `don't want X` create a `REMOVE`.
2. `NeedModel.revise()` collects every negative `(slot, value)` from the same turn before applying
   positives. This also prevents a later LLM fallback call from resurrecting a value rejected by the
   deterministic parser.

The result is an active `polyester` preference and an auditable rejected `linen`, never simultaneous
positive and negative linen.

### 4.6 Cascade clearing on category changes

A product category creates a semantic branch. Modifiers that described the old branch should not
automatically describe the new one:

> “Linen shirt... wait, no, running shoes.”

Without cascade clearing, the new query became `shoe linen running`, which favored linen/canvas
loafers. In `NeedModel.revise()`, a positive category `SET` compares the new category with the prior
category. On a switch it deactivates positive events in `CATEGORY_MODIFIER_SLOTS` that were learned
before the switch:

- material;
- color;
- style;
- feature;
- brand;
- size; and
- budget.

Span-aware logic handles two categories inside one noisy turn, while turn metadata handles switches
across turns. `use_case` is retained generally because a use case can cross closely related category
wording, but explicit footwear transitions contain additional rules:

- `boot -> shoe` clears `hiking` from the abandoned boot branch;
- a new positive `running` while the current category is `boot` infers a `shoe` category switch; and
- noisy mentions such as `running kind` or `running does` can therefore repair the state even when
  “shoes” is omitted or mistyped.

### 4.7 Implicit rejection of a recommendation

Users often criticize the recommendation instead of using a formal negation:

> “Why loafers, bro? I said running shoe.”

`_IMPLICIT_REJECTION_RE` recognizes complaint-plus-redirection patterns involving `why`, `instead`,
`give me`, `showing me`, `recommending me`, `want`, or `need`. The parser emits
`CLEAR(__modifiers__)`, then retains the positive category/use-case facts from the replacement part.

Category nouns mentioned as the *wrong recommendation*—for example “why are you giving me snow
boots?”—are also filtered from category extraction so the complaint cannot become a new boot request.

### 4.8 Boundary answers

`NO_PREFERENCE_RE` recognizes phrases such as:

- “I don't have a preference for color”;
- “no color preference”;
- “color doesn't matter”; and
- “whatever color is fine.”

The slot is added to `NeedModel.no_preference`. `Agent.respond()` projects that set into
`ConversationState.boundary_attrs`, and both `QuestionSelector.select()` and `next_ask()` skip it.
This prevents clarification loops and preserves the user's explicit boundary.

---

## 5. Dual-Track Query Builder

The query builder lives in `ConversationState.query_text()` in
[`src/dialogue.py`](../src/dialogue.py). Raw dialogue remains stored in `all_text` for audit and response
generation, but it is not automatically search text.

### 5.1 Clean track: effective-state projection

When `leaky_evidence == False`, the query is built from:

1. active positive ledger events, deduplicated in ledger order;
2. the canonical current category, placed first as the anchor;
3. an optional category anchor that is still compatible with the active category;
4. active, sanitized phrase-level constraint disclosures; and
5. safe unparsed terms from only the latest active sentence.

The builder never concatenates the full transcript on this path. `NeedModel.excluded_terms()` returns
rejected and superseded event values that have no active positive equivalent. Those terms are removed
from both phrase-level constraints and the natural-language fallback.

For the earlier noisy example, the effective clean query is conceptually:

```text
hat yellow
```

not:

```text
fluffy slipper bucket red blue yellow ...
```

### 5.2 Unparsed natural-language fallback

Offline evaluation disables LLM slot extraction, and deterministic vocabularies cannot enumerate
every useful phrase. Descriptions such as “rich napped pile” must still reach retrieval.

`ConversationState._unparsed_active_terms()` provides a bounded fallback:

- it reads only the latest turn, not accumulated history;
- if a repair cue exists, it keeps only the suffix after the last cue;
- it removes evaluator marker payloads already represented in the phrase ledger;
- it removes current structured event surfaces;
- it removes all `excluded_terms()`;
- it removes conversational filler and generic words; and
- it keeps otherwise-unparsed descriptive tokens for BM25/dense retrieval.

This improves recall without reopening append-only query leakage.

### 5.3 Phrase-ledger synchronization

The evaluator can disclose exact catalog constraints after markers such as “a key requirement is:”.
These live in `ConversationState.constraint_phrases` because they may not map cleanly to a fixed slot.

Two methods keep them consistent with the structured ledger:

- `effective_constraint_phrases()` removes rejected/superseded values from every active phrase.
- `invalidate_historical_phrases(turn)` retires disclosures from earlier turns after a category
  switch, repair cue, or modifier clear.

Without this synchronization, an old phrase could disappear from `NeedModel` while continuing to
demote the correct target during coverage ranking.

### 5.4 Leaky compatibility track

The released public simulator can expose wording taken directly from target catalog fields, such as
`Imported`, `100% Cotton`, `Button closure`, material labels, or branded/catalog-native phrases. The
old baseline benefited from sending raw accumulated text to BM25. The clean ledger projection is
better conversation modeling, but can discard raw wording that is useful for this evaluation
distribution.

The final system preserves both behaviors:

```python
if state.leaky_evidence:
    query = " ".join(state.all_text)
else:
    query = clean_ledger_projection
```

This switch is based on evidence, not on a public/private flag.

### 5.5 Immediate Turn 1 detection

`Agent._detect_turn1_leak()` runs before parsing and retrieval in `Agent.respond()`. It:

- isolates text after a known constraint marker when present;
- strips an ordinary “I'm looking for category” lead-in otherwise;
- extracts real contiguous stopword-free bi-grams and tri-grams;
- probes a bounded number of those n-grams against the catalog;
- recognizes field labels and boilerplate such as material/fabric labels, `Imported`, closure terms,
  percentage compositions, and catalog brand overlap; and
- requires multiple corroborating matches unless a stronger field-labelled signal is present.

If this detector fires, Turn 1 itself uses raw mode. There is no one-turn latency penalty.

### 5.6 Later-turn evidence and two confidence levels

`Agent._retrieve()` starts with the current query, then uses `_leaky_exact_match_count()` to probe
active disclosed phrases against both the current candidate head and bounded phrase-specific BM25
heads. It maintains two related flags:

| Flag | Threshold and effect |
|---|---|
| `leaky_evidence` | One complete catalog-backed phrase is enough to rerun retrieval with raw accumulated history. This is a recall decision. |
| `leaky_ranking_evidence` | Requires stronger evidence: multiple complete phrases plus catalog-native metadata vocabulary, or a strong Turn 1 signal. This enables the high popularity prior. |

The split is deliberate. A weak exact coincidence can justify a retrieval probe, but should not
activate every aggressive ranking prior.

---

## 6. Retrieval architecture

### 6.1 BM25 base

`Catalog` in [`src/catalog.py`](../src/catalog.py) loads the frozen JSONL catalog into:

- an in-memory product dictionary; and
- an in-memory SQLite FTS5 index.

BM25 searches title, categories, features, details, store, and description with field weights from
[`src/config.py`](../src/config.py). Query terms are normalized, deduplicated, capped, and joined as an
OR expression. No external vector database or mutable catalog is required.

### 6.2 Optional dense BGE route

`VectorRetriever` in [`src/retrieval.py`](../src/retrieval.py) uses cached BGE embeddings when they are
available. On the clean path with active constraints, the configured target ratio is:

```text
BM25  = 0.25
dense = 0.75
```

Because the RRF implementation treats BM25 as the route with weight `1.0`, `Agent._retrieve()`
converts the ratio to an equivalent dense secondary weight of `0.75 / 0.25 = 3.0`. The leaky path
keeps its intent/DCP-selected route weight because exact lexical retrieval is valuable there.

If the embedding cache or dependency is unavailable, initialization catches the failure and the
agent continues with BM25. This graceful fallback is important for reproducible local judging.

### 6.3 Reciprocal-rank fusion and expansion

BM25 and dense result lists are fused by reciprocal-rank fusion (`rrf()`), which is robust to
incomparable score scales. A low-weight expansion side track adds:

- normalized active ledger values;
- synonym/implication expansions from `ExpansionTable`; and
- use-case implications such as running -> lightweight/breathable/cushioned.

Expansion is always derived from the effective query and active positives. It does not resurrect
superseded raw transcript terms.

### 6.4 Stable candidate pool

The normal retrieval pool is 200 candidates. `Agent.POOL_SIZE_OVERRIDE` lets evaluation pin that
size independently of personalization or popularity. This fixed a measurement bug where the
popularity ablation accidentally used the legacy `POOL_NO_PERSONALIZATION = 10`, conflating a ranking
ablation with a 200-to-10 recall collapse.

---

## 7. Unified Multi-Stage Ranker

The production scorer is `DualTrackRanker.rank()` in [`src/ranking.py`](../src/ranking.py). It receives
the fused retrieval order and computes one additive final score for every candidate.

### 7.1 Final score

For candidate `c`:

```text
Final(c) =
    w_ret      * RetrievalRank(c)
  + w_sat      * Satisfaction(c)
  + w_cov      * ExactCoverage(c)
  + w_legacy   * LegacyCoverageRank(c)
  + w_cum      * CumulativeCoverage(c)
  +               RawNGramBonus(c)
  + w_pop      * Popularity(c)
```

All component weights are clamped non-negative. Ties fall back to the incoming retrieval order,
making the result deterministic.

The current core settings in `src/config.py` are:

| Signal | Current setting | Role |
|---|---:|---|
| Retrieval rank | `1.8` | Preserves the hybrid retrieval consensus |
| Satisfaction | `1.0`, or `2.0` on a constrained clean turn | Promotes products satisfying active needs |
| Discriminating exact coverage | `2.5` when gate fires, otherwise `0.0` | Strong exact-match promotion only when earned |
| Leaky exact coverage fallback | `4.0` | Restores public exact-coverage behavior after multiple complete matches |
| Legacy coverage-order rank | `3.0` | Blends the successful historical public ordering |
| Cumulative exact coverage | `5.0` | Rewards matching several active short constraints |
| Raw n-gram bonus | `0.2` per current-turn match | Breaks cumulative-coverage ties |
| Ordinary popularity | `0.10` | Small prior for vague turns |
| Strong leaky popularity | `1.0` | Public-compatible final tie resolution after strong evidence |

These constants are deliberately centralized in `src/config.py` so experiments do not hide policy
inside ranking code.

### 7.2 Retrieval rank signal

The incoming candidate order is converted to a normalized linear rank score from `1.0` for the first
candidate to `0.0` for the last. This preserves the evidence from BM25/dense RRF even though those
retrievers have different native score scales.

This term was added after the oracle showed that the target was usually already shallow in the
candidate pool but was being ranked out of the visible Top 10.

### 7.3 Need satisfaction

`NeedSatisfactionScorer.score_map()` generalizes lexical coverage. For each active phrase, candidate
agreement is:

```text
max(IDF-weighted lexical fraction, semantic_alpha * dense cosine)
```

Phrase scores are length-weighted and averaged. If vector support is unavailable, the semantic term
is absent and the scorer degrades to lexical satisfaction.

Critically, absent metadata contributes no negative term. In `DualTrackRanker.rank()`, a candidate
with neither satisfaction nor exact evidence receives a neutral `0.5` satisfaction floor. It is
treated as unknown and remains carried by retrieval rather than being classified as a mismatch.

On clean turns with at least one active ledger/phrase constraint, `Agent.respond()` raises the
satisfaction weight to `DUAL_CLEAN_W_SATISFACTION = 2.0`.

### 7.4 Discriminating exact-coverage gate

`DualTrackRanker._coverage_gate()` answers: *does exact phrase coverage actually identify one
candidate, or is it shared boilerplate?*

It computes:

- complete exact-phrase count for the top exact candidate;
- discrimination against a strong rival near the 90th percentile; and
- the fraction of the pool within 80% of the top exact score.

The high exact weight activates only when:

```text
complete matches >= 2
discrimination >= 0.35
shared fraction <= 0.35
```

Thus a distinctive exact disclosure can dominate, while weak or flat exact evidence leaves the
semantic/retrieval path in control.

### 7.5 Legacy coverage-order compatibility

Analysis of `origin/main` showed that the old public strength came from sorting aggressively on raw
verbatim coverage, blending popularity into that coverage score, and using retrieval order as a
bounded floor. That mechanism performed well on exact catalog-derived disclosures but collapsed on
paraphrases.

The final ranker reuses `CoverageReranker.rerank_scored()` only after at least two complete active
phrases match one candidate. Its resulting order is converted into another normalized rank feature,
`LegacyCoverageRank`, and blended into the unified score. It does not replace the P1 ledger or the
clean retrieval route.

### 7.6 Cumulative exact coverage

A strict “unique four-word phrase” override failed because many public misses contained several
short, shared phrases rather than one unique long phrase—for example:

```text
polyester; Imported; Button closure
```

`CoverageReranker.cumulative_exact_scores()` instead computes:

```text
CumulativeCoverage(c) =
    number of distinct active values exactly present in c
    ----------------------------------------------------
              number of distinct active values
```

Values come from active positive ledger constraints plus active phrase-ledger disclosures. Matching
uses normalized punctuation and true token boundaries, so `red` does not match `redwood`. The score
is bounded in `[0, 1]`; no active constraints yields exactly zero.

### 7.7 Raw n-gram tie-breaker

Cumulative coverage created large ties when dozens of products shared the same boilerplate. The
correct target could match 3/3 constraints and still sit behind many other 3/3 candidates.

`CoverageReranker._raw_ngrams()` and `raw_ngram_bonus_scores()` break those ties using exact,
contiguous bi-grams and tri-grams from the **current user message only**:

- stopword-containing windows are discarded rather than stitched together;
- when a constraint marker exists, only its payload is considered;
- punctuation is normalized through the shared tokenizer; and
- each distinct matching n-gram adds the configured fractional bonus.

Using only the current turn limits conversational leakage, while preserving title/metadata wording
that the structured parser intentionally ignores.

### 7.8 Conditional popularity

Popularity is useful as a prior under ambiguity and as the final microscopic tie-breaker among exact
public lookalikes, but it can bury an obscure correct target on an honest, specific request.

`Agent.respond()` therefore sets:

- `w_pop = 0` immediately on a clean turn with at least one active constraint;
- `w_pop = 0.10` on an otherwise vague ordinary turn; and
- `w_pop = 1.0` only when `leaky_ranking_evidence` is strong.

Popularity uses normalized `log1p(rating_number)`, not raw review count.

### 7.9 Bounded retrieval-head protection

On turns with no complete exact phrase, `_guard_retrieval_head()` guarantees that the first eight
hybrid-retrieval candidates remain somewhere in the visible Top 10. It replaces the lowest
unprotected fused candidates, then preserves final-score order within the selected head.

This is not a second ranker. It is a bounded-demotion safety net for the observed failure mode where
a target retrieved at rank 1–8 was pushed below rank 10 by noisy absolute satisfaction scores. Any
complete exact evidence disables the guard, so it does not dilute the public exact path.

### 7.10 Evaluator reveal contract

`Agent._reveal_count()` always returns the requested full list when `top_k >= 10`. Interactive calls
with a smaller `top_k` may still use adaptive reveal, but the official evaluator cannot mistake a
rank-2 pool candidate for a miss because only one item was shown.

---

## 8. Clean and leaky paths compared

| Decision | Clean / leak-free path | Catalog-exact compatibility path |
|---|---|---|
| Detection | No sufficient exact catalog evidence | Runtime Turn 1 or later catalog-backed evidence |
| Query | Active ledger + current category + active phrases + safe current-sentence fallback | Raw accumulated conversation history |
| Retrieval balance | 25% BM25 / 75% dense target when dense is available | Intent/DCP route weighting; lexical evidence retained |
| Satisfaction weight | Raised to `2.0` when constrained | Standard `1.0` |
| Exact coverage | Near zero unless discrimination gate earns it | Exact, legacy-order, cumulative, and n-gram evidence may activate |
| Popularity | Exactly zero once any active constraint exists | Strong prior only under stronger ranking evidence |
| Retrieval guard | Protects first eight retrieval candidates when exact evidence is absent | Disabled when complete exact evidence exists |
| Superseded terms | Strictly excluded | Raw-history compatibility intentionally preserves them after the evidence switch |

The last row is an explicit trade-off. Raw mode is not the desired model of a normal conversation;
it is a narrowly detected compatibility route for an evaluation distribution with direct catalog
wording. The clean path remains the default and the source of realistic correction behavior.

---

## 9. Personalization and memory safety

The optional Dynamic Context Programming layer is implemented in
[`src/context_engine.py`](../src/context_engine.py):

- `ContextDistiller` maintains a compact, recency-weighted session context and volatility trace.
- `ProfileService` stores time-decayed durable preferences.
- `OrchestrationPolicy` can emit per-turn route/pool/rerank plans.
- `GuidanceLearner` updates clarification weights from realized entropy/confidence changes.

### 9.1 Session state overrides durable state

The “stubborn boots” failure happened because durable `boot` and `hiking` tags continued to influence
personalization after the shopper repeatedly asked for running shoes.

`Agent._personalization_profile()` now projects the durable profile through the active ledger:

- every slot touched in the current session becomes authoritative;
- conflicting durable values for that slot are removed;
- rejected values are removed even when the durable item was stored as a generic `tag` without slot
  metadata;
- canonical category and known use-case tags are filtered against the active values; and
- all `NeedModel.excluded_terms()` are removed.

### 9.2 Corrections are persisted correctly

`ProfileService.write_through()` repeats the protection at storage time. It retires inactive or
rejected durable values before merging active positives using EMA/recency. A corrected
`boot + hiking -> shoe + running` session therefore writes shoe/running and cannot reintroduce the
old pair on the next reset.

### 9.3 Benchmark isolation

DCP is a product-facing optional layer, not a source of benchmark leakage. `new_isolated_agent()` in
[`scripts/eval_support.py`](../scripts/eval_support.py):

- creates a fresh `TemporaryDirectory` per evaluation agent;
- passes `persist_dcp=False`;
- disables all DCP feature flags on the instance; and
- retains the temporary-directory handle for the lifetime of the run.

This ensures that one evaluated session cannot teach or contaminate another. Normal application runs
may still opt into persistent profiles.

---

## 10. Dialogue, belief, and clarification

After ranking, `BeliefModel.update()` in `src/understanding.py` summarizes:

- the top candidate and its stability across turns;
- ranking margin and normalized entropy;
- current category;
- item confidence;
- need confidence; and
- uncertainty over missing required attributes.

`converge()` maps that belief to `PROBE`, `CONFIRM`, or `DELIVER`. `QuestionSelector.select()` chooses
the highest-value unresolved supported slot using uncertainty, category-aware decision weights, and
optional learned guidance.

The evaluator-facing action is deliberately aligned with the spoken question:

- if the selector asks about color, `ask_attribute` is `color`;
- if it asks about size, `ask_attribute` is `size`; and
- `other` is used only when no supported structured slot is available.

`next_ask()` never converts a concrete information-gain decision into `other`. It also skips slots in
`boundary_attrs` or `NeedModel.no_preference`, ensuring that a “no preference” answer is not asked
again.

This matters because the simulator uses `ask_attribute`—not merely the natural-language message—to
choose its next disclosure. A mismatch wastes turns and directly harms MTTC.

---

## 11. One complete turn, step by step

`Agent.respond()` is the orchestration boundary. A turn proceeds as follows:

1. **Load session state.** `reset()` must already have created a fresh `ConversationState`.
2. **Accumulate raw text for audit.** The message and turn number are appended to `all_text` and
   `message_turns`.
3. **Run the Turn 1 leak probe.** This happens before query construction so the first retrieval can
   use the correct track.
4. **Extract phrase disclosures.** `extract_constraints()` stores semicolon-delimited text after
   evaluator constraint markers with parallel turn metadata.
5. **Parse structured slots.** `SlotFiller.parse()` emits ordered ledger events. Optional LLM events
   use the same `Constraint`/`NeedModel.revise()` contract.
6. **Apply ledger revision.** Same-turn negations, exclusive overwrite, cascade clearing,
   no-preference boundaries, and special footwear repair rules resolve the active state.
7. **Synchronize phrase history.** Category switches, explicit repairs, and modifier clears retire
   old phrase-level evidence.
8. **Route intent.** `IntentRouter` updates a smoothed buying score and labels buying, browsing,
   mixed, or override behavior.
9. **Build the query.** `ConversationState.query_text()` chooses raw or clean projection.
10. **Retrieve.** BM25 runs first; later catalog-backed evidence may switch and rerun raw retrieval.
    Optional dense retrieval and expansion are fused into a pool.
11. **Build active ranking constraints.** Only positive, active, non-budget ledger values plus active
    phrase disclosures feed cumulative coverage.
12. **Rank once.** `DualTrackRanker.rank()` calculates every score component, sorts candidates, and
    applies the clean retrieval-head guard if eligible.
13. **Update belief and choose a question.** The output payload's `ask_attribute` stays synchronized
    with the selected slot.
14. **Write optional DCP state safely.** Corrected active state is authoritative; benchmarks disable
    this step.
15. **Reveal results.** Evaluator calls requesting ten items receive all ten valid recommendations.

---

## 12. Major technical challenges and their fixes

| Failure observed | Root cause | Final correction | Primary code |
|---|---|---|---|
| `I'm` became size `m`; `Valentine's` became size `s` | Python `\b` treats apostrophes as non-word boundaries | Negative apostrophe lookbehind/lookahead around single-letter size alternatives | `src/understanding.py` — `SIZE_RE` |
| Slippers, buckets, red, blue, and yellow coexisted | Append-only preference state | Ordered ledger with overwrite/remove/clear semantics and active versus audit views | `Constraint`, `NeedModel.revise()` |
| `instead of linen` produced both negative and positive linen | Generic material regex re-extracted the negated word | `_explicitly_negated()` plus same-turn negative set in `revise()` | `SlotFiller.emit()`, `NeedModel.revise()` |
| Durable `boot` and `hiking` overrode repeated running-shoe corrections | Profile prior was merged without active-session masking | Slot-aware profile projection and rejected-value purge at both read/rank and write-through | `Agent._personalization_profile()`, `ProfileService.write_through()` |
| Query still contained raw filler and abandoned history | Query builder concatenated conversation text | Clean ledger projection plus `excluded_terms()` | `ConversationState.query_text()` |
| “Rich napped pile” disappeared with LLMs off | Strict slots could not cover open vocabulary | Bounded latest-active-sentence fallback | `ConversationState._unparsed_active_terms()` |
| Old phrase constraints survived a category override | Structured and phrase state had separate lifecycles | Phrase turn metadata, sanitization, and historical invalidation | `effective_constraint_phrases()`, `invalidate_historical_phrases()` |
| Linen shirt -> running shoes returned linen loafers | Old category modifiers remained active | Span/turn-aware cascade clearing on category switch | `NeedModel.revise()` |
| “Why loafers?” did not change state | Complaint lacked an explicit `not` token | Implicit rejection emits modifier clear; recommendation nouns filtered as feedback | `_IMPLICIT_REJECTION_RE`, `SlotFiller.parse()` |
| Running-shoe correction stayed locked on boots | Footwear categories overlapped and typo/noisy phrasing omitted “shoe” | Running can infer boot -> shoe and clears hiking; complaint boot mention is ignored | `NeedModel.revise()`, category filtering in `SlotFiller.parse()` |
| Target at pool rank 2 still missed Top 10 | Adaptive reveal returned fewer than ten evaluator results | `top_k >= 10` always bypasses holdback | `Agent._reveal_count()` |
| Targets with missing metadata were demoted as mismatches | Absence was interpreted as failed satisfaction | Missing fields contribute no penalty; sparse candidates receive neutral evidence floor | `NeedSatisfactionScorer.score_map()`, `DualTrackRanker.rank()` |
| Strict unique four-word override did not lift public misses | Remaining disclosures were several shared short phrases | Cumulative exact coverage across all active values | `CoverageReranker.cumulative_exact_scores()` |
| Many candidates tied at 3/3 cumulative coverage | Shared Amazon boilerplate gave identical scores | Current-turn exact bi/tri-gram bonus, then conditional popularity | `_raw_ngrams()`, `raw_ngram_bonus_scores()` |
| Clean ranker pushed shallow retrieval hits below Top 10 | Absolute satisfaction noise overrode good relative retrieval | Retrieval feature plus bounded first-eight head protection | `DualTrackRanker._guard_retrieval_head()` |
| Clean query hurt public Turn 1 MRR/MTTC | Exact catalog evidence was detected only after later disclosures | Catalog-backed Turn 1 probe before first retrieval | `Agent._detect_turn1_leak()` |
| Popularity ablation appeared to destroy retrieval | It accidentally reduced pool size from 200 to 10 | Independent pool override and popularity controls | `Agent._pool_size()`, `scripts/eval_matrix.py` |
| Evaluation sessions contaminated one another | Persistent DCP store reused across sessions/runs | Fresh temporary directory and DCP disabled per benchmark agent | `scripts/eval_support.py` |
| Generated stress sets did not exactly match the official mix | Naive rounding of finite proportions | Deterministic largest-remainder allocation | `scripts/scenario_mix.py` and both set builders |
| Bot voiced a specific question but sent `ask_attribute="other"` | Display phrasing and structured action diverged | Specific information-gain slot is authoritative in payload; `other` is last resort | `QuestionSelector.select()`, `next_ask()` |

---

## 13. Evaluation architecture and evidence

### 13.1 Official metrics

The local evaluator in [`evaluator/local_evaluator.py`](../evaluator/local_evaluator.py) uses:

```text
HitRate@10 = sessions where target first appears in the returned Top 10 / sessions
MRR        = mean reciprocal rank at first hit
MTTC       = mean first-hit turn, with misses assigned turn 11
Efficiency = clip((11 - MTTC) / 10, 0, 1)

TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
```

The evaluator ends a session at the first valid Top-10 hit. Therefore first-turn rank quality,
structured clarification actions, and full-list reveal directly influence MRR and MTTC.

### 13.2 Honest and public sets

The project intentionally measures two distributions:

- `data/language_stress_set.jsonl` is the primary leak-free language stress set. Catalog evidence is
  reworded so exact copied phrases cannot carry the system.
- `data/public_set.jsonl` is the official public guardrail. Its simulator may surface exact target
  catalog wording, which is why exact coverage remains a compatibility requirement.
- `data/pillar_free.jsonl` supports per-pillar diagnosis.

`scripts/build_language_stress_set.py` and `scripts/build_pillar_sets.py` use
`scripts/scenario_mix.py` to allocate exact finite quotas with the largest-remainder method:

| Scenario | Share |
|---|---:|
| Buying | 40% |
| Browsing | 40% |
| Intent override | 15% |
| Boundary | 5% |

### 13.3 Diagnostic scripts

- `scripts/eval_matrix.py` reports `score / hit@10 / MRR / MTTC` for the active dual-track ranker in
  normal and popularity-ablated modes. Pool size stays fixed in both.
- `scripts/oracle_leakfree.py` separates retrieval misses from ranking misses by checking whether the
  target ever entered the full candidate pool.
- `scripts/trace_eval.py` and the opt-in tracer in `src/trace.py` show query, track flags, pool,
  fusion details, belief, selected question, and target rank per turn.
- `scripts/chat.py` exposes interactive `:state` inspection of the ledger, effective query, DCP
  state, and current plan.

### 13.4 What the measurements taught us

The most important oracle result during development was:

```text
retrieval recall: 243 / 250 = 97.2%
end-to-end Hit@10 at that checkpoint: 193 / 250 = 77.2%
ranking-fault misses: 50 / 57 = 87.7%
median target pool rank on ranking misses: 18
```

This proved that widening retrieval further was not the highest-value next action. The target was
usually already in the 200-item pool; final fusion was suppressing it. That evidence motivated the
retrieval feature, satisfaction changes, sparse-listing neutrality, cumulative coverage, n-gram
tie-breaker, and bounded retrieval-head guard.

Development checkpoints from the branch's recorded runs show the progression. These are local
measurements, not hard-coded guarantees; rerun them in the target environment before quoting a final
submission number.

| Checkpoint | Leak-free result | Public result | Interpretation |
|---|---|---|---|
| Original honest diagnosis | TechnicalScore about `0.3825` | High but leak-dependent | Browsing and paraphrase behavior were weak |
| Correction-aware P1 + improved recall | pool recall `97.2%`, Hit@10 `67.2%` | Public declined | State/query correctness exposed ranking as the bottleneck |
| P2 semantic-preserving ranking | TechnicalScore about `0.7157`, Hit@10 `81.6%` | Below the `0.8800` guardrail initially | Honest ranking recovered substantially |
| Guardrail-tuned dual track | TechnicalScore about `0.7240` | about `0.8886` | Clean popularity deletion plus evidence-gated public compatibility balanced both tracks |
| Latest full run after structured questioning | about `0.7366` (`Hit@10 0.868`, `MRR 0.513`, `MTTC 3.57`) | about `0.8816` (`Hit@10 0.995`, `MRR 0.745`, `MTTC 2.98`) | Public remains above the required `0.8800`; clarification semantics improve while preserving the guardrail |

The central result is architectural rather than one number: clean conversational corrections no
longer depend on stale raw history, while exact public evidence is recovered through an independently
detected runtime path.

---

## 14. Testing strategy

### 14.1 Unit and component tests

[`tests/test_correction_ledger.py`](../tests/test_correction_ledger.py) is the regression suite for:

- exclusive overwrite and explicit conjunction;
- repair cue ordering;
- negative-versus-positive precedence;
- category cascade clearing and boot-to-shoe repair;
- recommendation-complaint parsing;
- no-preference boundaries;
- profile masking and write-through retirement;
- strict effective-query exclusion;
- raw-query activation only after marking;
- the apostrophe/size boundary bug;
- unparsed active-language fallback;
- phrase-history invalidation; and
- full Top-10 evaluator reveal.

[`tests/test_components.py`](../tests/test_components.py) covers ranking gates, cumulative matching,
token boundaries, n-gram tie-breaking, strong/weak leaky evidence, scenario allocation, fixed-pool
ablations, persistence isolation, and core component behavior.

[`tests/test_evaluator.py`](../tests/test_evaluator.py) checks the official local evaluation contract and
metric behavior.

Run:

```bash
python -m pytest -q
```

### 14.2 Recommended evaluation sequence

```bash
# Current primary/guardrail matrix, including MTTC
python -u scripts/eval_matrix.py --public-n 200 --pool-size 200

# Determine whether remaining misses are retrieval or ranking failures
python -u scripts/oracle_leakfree.py

# Manual adversarial conversation inspection
python -u scripts/chat.py
```

For manual testing, use messages that exercise state transitions rather than clean slot lists:

```text
I want cotton—actually linen. Wait no, running shoes like hiking boots, but waterproof.
Why loafers? I said running shoe.
Instead of linen I want polyester.
Color doesn't matter.
```

After each turn, inspect `:state` and verify:

1. only the intended category and modifiers are active;
2. rejected/superseded ledger entries remain auditable but inactive;
3. the effective clean query contains no abandoned value;
4. the profile contains no conflicting active influence;
5. `ask_attribute` names the slot actually being asked; and
6. evaluator mode returns ten valid ASINs.

### 14.3 How to diagnose a miss

Use this order:

1. **State failure:** Is the target request correctly represented in active ledger constraints?
2. **Query failure:** Does `query_text()` include active evidence and exclude retired evidence?
3. **Track failure:** Did `leaky_evidence` or `leaky_ranking_evidence` fire appropriately?
4. **Retrieval failure:** Is the target absent from the 200-item pool on every turn?
5. **Ranking failure:** If present, which fusion component pushed it below Top 10?
6. **Reveal failure:** Was it ranked in the visible head but not returned?
7. **Dialogue failure:** Did the action request a useful supported slot, and did the simulator reply
   to that same slot?

This sequence prevents treating every miss as a retrieval problem or changing multiple independent
mechanisms at once.

---

## 15. Configuration and feature discipline

`Agent` class attributes in `src/agent.py` are the feature-flag ledger. Their comments distinguish:

- **core measured paths**, enabled by default;
- **optional/unproven product features**;
- **demo-only features**; and
- **measured neutral or negative ablations** retained for research.

Notable defaults:

- Dual-track ranking is on.
- Vector retrieval is attempted but automatically disables itself if unavailable.
- LLM slot/use-case helpers may be configured for product use, but all benchmark scripts turn LLM
  slots, inference, response generation, and LLM reranking off.
- MMR diversity is off on the scored path because moving tail items can hurt a single-target metric;
  it remains available for a more varied live demo.
- DCP is product-facing and on in ordinary runs, but explicitly off and non-persistent in evaluation.

This keeps the submitted behavior explainable and allows individual signals to be ablated without
silently changing pool size or unrelated state.

---

## 16. Limitations and future work

1. **Leak detection is heuristic.** It is deliberately conservative, but a normal shopper using
   catalog-native wording can activate raw mode, and a subtle copied phrase can remain undetected.
   A calibrated classifier using only catalog-frequency features would be a cleaner future design.
2. **Category repair contains domain-specific rules.** The running/boot/shoe logic solves a measured
   footwear failure. A general ontology of mutually exclusive product branches and modifier scope
   would scale better across all categories.
3. **Deterministic slot vocabularies are finite.** The safe unparsed fallback preserves recall, but
   does not provide the same explainability as a structured slot. More catalog-derived vocab mining
   could expand coverage without requiring an LLM.
4. **Dense retrieval depends on local artifacts.** The architecture supports BGE, but environments
   without the embedding cache run BM25-only. Final benchmark reporting should state which path ran.
5. **Public guardrail and real conversation are different distributions.** Raw-history compatibility
   is intentionally isolated, but it remains a benchmark-specific adaptation. Product deployment
   should monitor how often it activates and may choose to disable it.
6. **Clarification optimizes an imperfect simulator.** Structured questions are semantically correct,
   yet the official simulator's disclosure policy may reward generic prompts differently. Per-pillar
   MTTC analysis remains important.
7. **DCP effectiveness is not yet fully ablated.** The memory layer is safety-correct and isolated in
   benchmarks, but its product value should be measured separately on repeated-user sessions.

---

## 17. Suggested README, Devpost, and demo narrative

### 17.1 One-sentence pitch

> A correction-aware shopping copilot that turns messy, self-revising conversation into an auditable
> preference ledger, searches through a semantic/exact dual track, and ranks with evidence-adaptive
> fusion instead of blindly accumulating keywords.

### 17.2 Technical story for the pitch

1. **Show the failure:** an append-only agent remains stuck on boots, hiking, or linen after the user
   changes their mind.
2. **Show the ledger:** display old entries becoming inactive while `shoe + running` remains active.
3. **Show the effective query:** contrast noisy raw history with the clean query.
4. **Show category cascade:** switch from a linen shirt to running shoes and demonstrate that linen
   does not produce loafers.
5. **Show negation safety:** “instead of linen, polyester” leaves linen rejected even if an LLM
   fallback tries to add it.
6. **Show ranking evidence:** explain that retrieval recall reached 97.2%, proving the remaining work
   belonged in fusion rather than another retrieval rewrite.
7. **Show dual-track balance:** paraphrased requests preserve semantic retrieval; exact catalog
   disclosures activate cumulative coverage and n-gram tie-breaking.
8. **Close with measurement:** run the isolated matrix and report Hit@10, MRR, and MTTC for both the
   leak-free primary set and public guardrail.

### 17.3 What makes the architecture distinctive

- It models correction as a first-class state transition, not another positive keyword.
- It scopes modifiers to product-category branches.
- It propagates rejection consistently through state, query, phrase coverage, and durable memory.
- It uses exact-match evidence adaptively instead of choosing permanently between semantic and
  keyword ranking.
- It diagnoses performance by separating pool recall from final-rank failure.
- It keeps evaluation scientifically honest through exact scenario composition, fixed pools, and
  isolated non-persistent sessions.

---

## 18. Implementation reference

| File / function | Responsibility |
|---|---|
| `src/understanding.py::Constraint` | Preference-ledger event schema |
| `src/understanding.py::NeedModel.revise` | Ordered operation application, negation priority, overwrite, cascade clearing, footwear repair |
| `src/understanding.py::NeedModel.excluded_terms` | Rejected/superseded query mask |
| `src/understanding.py::SlotFiller.parse` | Regex slots, repair cues, implicit complaints, NO_PREFERENCE/CLEAR events |
| `src/understanding.py::SIZE_RE` | Apostrophe-safe size extraction |
| `src/understanding.py::QuestionSelector.select` | Belief/information-gain clarification selection |
| `src/dialogue.py::extract_constraints` | Phrase-level evaluator disclosures |
| `src/dialogue.py::ConversationState.query_text` | Dual-track raw versus clean query projection |
| `src/dialogue.py::ConversationState._unparsed_active_terms` | Safe open-vocabulary clean-query fallback |
| `src/dialogue.py::ConversationState.effective_constraint_phrases` | Phrase sanitization against retired values |
| `src/dialogue.py::ConversationState.invalidate_historical_phrases` | Phrase lifecycle synchronization on corrections |
| `src/dialogue.py::next_ask` | Structured `ask_attribute` alignment and boundary skipping |
| `src/agent.py::Agent.respond` | End-to-end turn orchestration and conditional fusion weights |
| `src/agent.py::Agent._detect_turn1_leak` | Immediate catalog-shaped evidence detection |
| `src/agent.py::Agent._retrieve` | Query-track switch, BM25/dense/expansion retrieval, RRF |
| `src/agent.py::Agent._personalization_profile` | Active-session masking of durable preferences |
| `src/agent.py::Agent._reveal_count` | Full evaluator Top 10 guarantee |
| `src/ranking.py::CoverageReranker` | Catalog text cache and exact/legacy/cumulative/n-gram evidence |
| `src/ranking.py::NeedSatisfactionScorer` | Lexical/semantic need agreement with sparse-data neutrality |
| `src/ranking.py::DualTrackRanker._coverage_gate` | Exact-evidence discrimination decision |
| `src/ranking.py::DualTrackRanker._guard_retrieval_head` | Bounded clean-track demotion protection |
| `src/ranking.py::DualTrackRanker.rank` | Unified final score and deterministic sort |
| `src/context_engine.py::ProfileService.write_through` | Correction-safe durable profile update |
| `scripts/eval_support.py::new_isolated_agent` | Fresh non-persistent benchmark environment and fixed pool |
| `scripts/scenario_mix.py::largest_remainder_counts` | Exact official scenario quota allocation |
| `scripts/eval_matrix.py` | Current score/Hit/MRR/MTTC matrix and independent popularity ablation |
| `scripts/oracle_leakfree.py` | Retrieval-versus-ranking fault attribution |

This map is the recommended starting point for future maintenance: changes to preference semantics
belong in `NeedModel`, changes to query projection belong in `ConversationState`, changes to evidence
fusion belong in `DualTrackRanker`, and dataset/evaluation controls belong under `scripts/`.
