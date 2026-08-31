# Devpost Submission — Paste-Ready Text

Replace `[YOUR YOUTUBE URL]` with the public video link before submitting.

---

**Title:** TokenMaxx Copilot

**Tagline:** Offline-first conversational search that turns changing preferences into ranked product recommendations.

---

## Inspiration

Product search works well when a shopper already knows the right keywords. It is much less useful when the shopper says "I'm still exploring," describes an occasion instead of a product, has no preference for a requested attribute, or changes direction halfway through the conversation. A useful shopping assistant needs to preserve what is still true, retire what is no longer true, ask for information that can narrow the search, and remain reliable when external AI services are unavailable.

Conversational search can reduce the vocabulary gap between how people describe needs and how catalogs describe products, while making recommendations easier to inspect and correct.

---

## What it does

TokenMaxx Copilot searches a frozen 50,000-product clothing, shoes, and jewelry catalog over a conversation of up to ten turns. On each turn it can ask one structured follow-up question, return ranked product IDs, or do both.

The agent:
- Recognizes buying, browsing, boundary, and intent-override language
- Maintains a correction-aware preference ledger instead of blindly trusting the full transcript
- Retrieves candidates with BM25 and optionally a local dense embedding model
- Expands use-case and preference signals at a controlled weight
- Reranks candidates by how well they satisfy the active need
- Estimates confidence and decides whether to clarify or reveal more results
- Explains its top recommendation using catalog evidence
- Falls back to a fully deterministic path when optional models are unavailable

Representative verified flow:

```
Turn 1 shopper: I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.
Turn 1 agent: asks for another differentiating preference and reveals one candidate.
Turn 2 shopper: reveals a distinctive product feature.
Turn 2 agent: returns the correct product at rank 1, with 0 model tokens used.
```

---

## How we built it

`starter/agent.py` exposes the required Agent interface. `src/agent.py` orchestrates a turn across:

- **`src/catalog.py`** — JSONL loading, text normalisation, in-memory SQLite FTS5 index
- **`src/dialogue.py`, `src/understanding.py`, `src/belief.py`** — intent routing, correction-aware state, slot extraction, confidence, clarification, and rationale generation
- **`src/retrieval.py`** — optional BGE dense embeddings and reciprocal-rank fusion
- **`src/ranking.py`** — coverage scoring, need satisfaction, profile/quality priors, retrieval safeguards
- **`src/reranker.py`** — optional cross-encoder, Gemini listwise reranker, and experimental linear LTR
- **`src/context_engine.py`** — session distillation, persistent user profiles with time decay, and online guidance learning
- **`src/trace.py` + `app/`** — structured turn traces and a local conversation-inspection UI

Each optional component fails safely to the deterministic path. The verified public evaluation run used zero model tokens and no API key.

**Development tools:** Python 3.10+, pytest, Ruff, compileall, Git, Flask (trace UI)

**APIs and models:**
- SQLite FTS5 BM25 — required lexical retrieval
- BAAI/bge-small-en-v1.5 — optional local query/product embeddings (Hugging Face)
- cross-encoder/ms-marco-MiniLM-L-6-v2 — optional local precision reranker (Hugging Face)
- Google Gemini (`gemini-flash-lite-latest` via `google-genai`) — optional slot extraction, use-case inference, and response generation; credentials are environment-only, never committed

**Libraries:** Python standard library (`sqlite3`), NumPy, scikit-learn, sentence-transformers, google-genai, Flask, python-dotenv

**Dataset:** Amazon Reviews 2023 by McAuley Lab, UCSD — `Clothing_Shoes_and_Jewelry` category. Text and structured metadata only; no raw user identifiers, reviews, or product images. Full attribution in `DATA_ATTRIBUTION.md`.

---

## Challenges we ran into

- Separating genuine language understanding from the public simulator's target-derived wording
- Balancing target recall, rank, and first-hit timing when revealing results early can freeze MRR
- Handling preference corrections and intent overrides without reintroducing retired terms
- Keeping optional dense, cross-encoder, and Gemini components from making startup fragile
- Preventing durable profile state from contaminating independent evaluation sessions

---

## Accomplishments we're proud of

- A fully deterministic fallback with no required API key and zero tokens on the verified run
- 138 passing automated tests and a stable evaluator-facing API
- Public evaluation: Hit Rate@10 **0.965**, MRR **0.852**, MTTC **2.905**, TechnicalScore **0.9001**
- Honest-set diagnostic (250 sessions rewritten to avoid product-description phrasing): Hit Rate@10 **0.908**, MRR **0.664**, TechnicalScore **0.8071** — a self-imposed generalization audit showing the pipeline holds up when customers use their own language
- A `GuidanceLearner` that measures realized information gain per clarification question and reweights future question priorities online — dormant in offline evaluation, active in production
- An MMR result diversifier that surfaces distinct styles and gives smaller vendors fairer visibility — implemented and tested, disabled for the competition benchmark for a principled reason (explained in the repo)

---

## What we learned

The biggest lesson was the difference between optimizing a benchmark and building something useful. The public simulator's phrasing overlaps target product text in ways that inflate lexical retrieval scores — so we built our own 250-session honest set with rewritten language and ran the same pipeline against it with no changes. That discipline shaped every decision about what to ship on the scored path versus what to flag as a production-only feature.

---

## What's next

- Enable the GuidanceLearner and MMR diversifier in a live deployment and measure their effect on real conversion rates
- Run a human-written, target-independent language evaluation and usability study
- Add a deployment target, observability, session eviction, and concurrency controls
- Extend the correction-aware state ledger with typed records and broader override tests

---

## Impact and relevance

**Conversion friction** — MTTC 2.9 turns on a 10-turn budget. Every extra turn is a dropout risk; fewer turns means less friction between intent and purchase.

**Accessibility for small retailers** — the full scored path runs on SQLite with no API key and no cloud dependency. A retailer with 50,000 SKUs can deploy conversational search on modest infrastructure with $0 operating cost on the core path.

**Live commerce and discovery-first shopping** — in social commerce contexts where customers arrive from a video with a vague impression rather than a specific query, the system's dual buying/browsing routing and exploratory clarification strategy are the right primitives.

**A system that learns which questions to ask** — the GuidanceLearner measures which clarification questions actually reduce uncertainty and reweights them over time. A store running this for a month will ask better questions than on day one.

**Vendor fairness** — the MMR diversifier prevents ten near-identical items from the same dominant brand filling every result list, giving smaller vendors a fairer chance to appear in the visible window.

---

## Links

- Repository: https://github.com/landonzhao/techjam-conversational-search
- Architecture: https://github.com/landonzhao/techjam-conversational-search/blob/main/architecture.md
- Demo video: [YOUR YOUTUBE URL]

---

## Team

- **Landon Zhao** — Modular conversational-search architecture, hybrid retrieval and ranking, dialogue strategy, dynamic conversation policy, session context, intent-override handling, and ranking robustness evaluation
- **Valerie Lim** — Satisfaction-ranking and leak-free retrieval improvements, regression and experiment tooling, integration, final repository stabilization, validation, documentation, and submission preparation
- **Bryan Koh** — Correction-aware state ledger, evaluator isolation and retrieval optimisations, dual-track ranking and coverage mechanisms, structured clarification behavior, and contradiction/negation handling
