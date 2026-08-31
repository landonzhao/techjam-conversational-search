# Devpost Project Story — Paste-Ready

---

## Inspiration

Most product search fails the moment a shopper doesn't know the right keyword.

The shopper describes an occasion: "something for a beach wedding."
Or changes their mind halfway: "actually, not gold, I want silver."

And keyword search has no answer.

We (jollibee888) wanted to build an agent that handles this the way a good sales associate would: tracking what you said, forgetting what you retracted, and asking the one question most likely to help find what you're actually looking for.

The deeper motivation was measurement honesty. The competition's scoring formula is:

$$\text{TechnicalScore} = 0.50 \times \text{HitRate@10} + 0.30 \times \text{MRR} + 0.20 \times \text{Efficiency}$$

Where Efficiency rewards finding the target in fewer turns.

We noticed early that the public evaluator generates customer messages containing phrases lifted almost verbatim from the target product's own catalog description. This means a well-tuned lexical retriever can score very high without truly understanding natural language.

So we decided to build a system that wins both: the benchmark, and a separate 250-session honest set where customers describe needs in their own words — not catalog terms — to measure what the pipeline actually generalises to.

---

## What it does

TokenMaxx Copilot searches a 50,000-product clothing, shoes, and jewelry catalog over a conversation of up to ten turns. On each turn it can ask one structured follow-up question, return ranked product recommendations, or do both.

- **Remembers when you change your mind** — maintains an explicit constraint ledger that retires old preferences on intent override, rather than blindly accumulating the full transcript
- **Asks smarter questions over time** — a `GuidanceLearner` measures how much each clarification question actually narrowed the search, and reweights future question priorities accordingly
- **Works without any API or cloud dependency** — the full scored path runs on SQLite and the Python standard library, $0 operating cost
- **Diversifies results for fairer coverage** — an MMR diversifier surfaces distinct styles instead of ten near-identical items from the same brand

Representative verified flow:

```
Turn 1 shopper: I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.
Turn 1 agent:   asks for one more differentiating preference, reveals one candidate.
Turn 2 shopper: reveals a distinctive product feature.
Turn 2 agent:   returns the correct product at rank 1. Model tokens used: 0.
```

---

## How we built it

The architecture is a staged pipeline with clean separation between components:

```
message + profile
  → intent and constraint-state update
  → BM25 retrieval (+ optional dense/expansion routes)
  → need-satisfaction ranking (+ optional cross-encoder)
  → belief, clarification, and reveal policy
  → contract-shaped response
```

**Retrieval** uses in-memory SQLite FTS5 BM25 as the always-on baseline, with an optional BGE dense embedding track fused via reciprocal-rank fusion. The two tracks are weighted differently for buying versus browsing intent.

**State** is maintained as an explicit constraint ledger — not a raw transcript. Every preference is stored with its polarity, turn, and weight. On intent override, old constraints are soft-demoted rather than evicted, because the evaluator constructs messages partly from the target's soft preferences — keeping a partial signal helps.

**Ranking** scores candidates by how well their catalog text covers the active constraint phrases. A local cross-encoder (`ms-marco-MiniLM-L-6-v2`) runs by default and is gated by a regime router: on turns where exact phrase matches are strong, the lexical coverage signal takes precedence; on turns where it is weak, the cross-encoder provides high-precision reranking. This prevents the cross-encoder from diluting the verbatim signal when it is already decisive.

**Self-evolution** is implemented via the `GuidanceLearner`: it measures the entropy drop in the belief model after each clarification question and reweights future question priorities with an exponential moving average. Dormant in offline evaluation (each session resets), active in a live deployment.

**Result diversity** is handled by an MMR diversifier that protects the confident top picks and fills remaining slots by penalising title-token similarity to already-selected items. Off by default for the competition benchmark, on for a real storefront.

---

## Challenges we ran into

**Separating benchmark performance from real-language performance.** The verbatim overlap between simulator phrasing and product descriptions inflates lexical retrieval scores in a way that would not hold for real shoppers. We built the honest set specifically to audit this gap — and used it to decide which components to ship on the scored path versus flag as production-only.

**The adaptive reveal problem.** Revealing results too early locks the target's MRR at whatever rank it holds on turn 1. Revealing too late costs Efficiency. We implemented a confidence-gated reveal policy that holds back a short list while belief is low, and reveals unconditionally from turn 4 onward. This alone moved MRR from 0.705 to 0.861.

**Intent override without reintroducing retired terms.** When a shopper says "forget the red, I want blue," naively appending to the constraint ledger leaves "red" in the retrieval query. We handle this with soft demotion — old phrases stay at reduced weight because they are still true of the target's soft preferences, just no longer the primary signal.

**Keeping optional components from making startup fragile.** Dense embeddings, the cross-encoder, and Gemini each sit behind a separate availability check. If any one fails to load, the system falls through to the next layer without raising. The verified public run used zero model tokens.

---

## Accomplishments that we're proud of

- Public evaluation: Hit Rate@10 **0.965**, MRR **0.852**, MTTC **2.905**, TechnicalScore **0.9001** — zero model tokens, no API key
- Honest-set diagnostic (250 sessions in customers' own words, not catalog terms): Hit Rate@10 **0.908**, MRR **0.664**, TechnicalScore **0.8071** — the same pipeline, no configuration changes
- A `GuidanceLearner` that learns which clarification questions narrow the search fastest — implemented and running, not a roadmap item
- An MMR diversifier for fairer result coverage — implemented, tested, and deliberately left off for the competition with a documented reason
- 138 passing automated tests, clean lint, and a fully reproducible evaluation path

---

## What we learned

Building the honest set was the most valuable thing we did. It forced every feature decision to answer one question: does this help because the pipeline is better, or because the simulator leaks?

The ~9-point gap between our public score (0.9001) and honest-set score (0.8071) is not a failure — it is an accurate measurement of how much work the verbatim signal is doing. The cross-encoder and structured-constraint tracks are the designed answer to closing it on the private set and in production.

The second lesson: the features that matter most for real users — result diversity, self-improving question selection, correction-aware memory — are all penalised or invisible in the benchmark. Knowing which flags to turn on for scoring versus which to turn on for a real storefront requires building both and measuring both.

---

## What's next for TokenMaxx Copilot

- Enable the `GuidanceLearner` and MMR diversifier in a live deployment and measure their effect on real conversion rates
- Run a human-written, target-independent language evaluation and usability study
- Add a deployment target, session eviction, observability, and concurrency controls
- Extend the correction-aware state ledger with typed records and broader override tests
