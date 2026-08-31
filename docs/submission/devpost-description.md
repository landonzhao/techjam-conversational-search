# Devpost Description Draft

## Project title and tagline

**TokenMaxx Copilot**

*Offline-first conversational search that turns changing preferences into ranked product
recommendations.*

## Inspiration and problem

Product search works well when a shopper already knows the right keywords. It is much less useful
when the shopper says “I’m still exploring,” describes an occasion instead of a product, has no
preference for a requested attribute, or changes direction halfway through the conversation. A
useful shopping assistant needs to preserve what is still true, retire what is no longer true, ask
for information that can narrow the search, and remain reliable when external AI services are
unavailable.

This matters beyond a benchmark. Conversational search can reduce the vocabulary gap between how
people describe needs and how catalogs describe products, while making recommendations easier to
inspect and correct.

## What it does

TokenMaxx Copilot searches a frozen 50,000-product clothing, shoes, and jewelry catalog over a conversation
of up to ten turns. On each turn it can ask one structured follow-up question, return ranked product
IDs, or do both.

The agent:

- recognizes buying, browsing, boundary, and intent-override language;
- maintains a correction-aware preference ledger instead of blindly trusting the full transcript;
- retrieves candidates with BM25 and optionally a local dense embedding model;
- expands use-case and preference signals at a controlled weight;
- reranks candidates by how well they satisfy the active need;
- estimates confidence and decides whether to clarify or reveal more results;
- explains its top recommendation using catalog evidence;
- reports model token usage and falls back to a deterministic path when optional models fail.

## How it addresses the challenge

The challenge rewards finding one hidden target product early and high in the top ten. TokenMaxx Copilot
maps each judging-relevant problem to a concrete component:

- **Recall:** field-weighted SQLite FTS5 BM25, optional BGE dense search, and reciprocal-rank fusion.
- **Precision:** lexical/semantic need-satisfaction scoring, retrieval safeguards, and an optional
  local cross-encoder.
- **Multi-turn reasoning:** per-session state, a structured update ledger, boundary handling, and
  soft demotion of pre-override phrases.
- **Conversation strategy:** belief/confidence updates, adaptive clarification, discovery wording,
  and adaptive reveal.
- **Feasibility:** a zero-credential standard-library fallback, optional model isolation, bounded
  candidate pools, and deterministic local evaluation.

## End-to-end user experience

1. The shopper begins with a product category, a hard constraint, or an exploratory statement.
2. TokenMaxx Copilot retrieves and ranks an initial candidate pool.
3. When confidence is low, it asks for one attribute or presents useful alternatives to compare.
4. The shopper adds, removes, or revises a preference.
5. The state ledger updates the active need, retrieval and ranking rerun, and the agent reveals the
   strongest current recommendations.
6. The interaction ends when the correct product enters the top ten or after turn ten.

Representative verified evaluator flow:

```text
Turn 1 shopper: I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.
Turn 1 agent: asks for another differentiating preference and reveals one candidate.
Turn 2 shopper: reveals a distinctive product feature.
Turn 2 agent: returns B09PYB7B6Z at rank 1, with 0 model tokens used.
```

The exact wording is generated from current candidates and state, so the demo should show the live
trace rather than hard-code a sentence.

## Technical implementation

`starter/agent.py` exposes the required `Agent`. `src/agent.py` orchestrates a turn across:

- `src/catalog.py`: JSONL loading, text normalization, and an in-memory SQLite FTS5 index;
- `src/dialogue.py`, `src/understanding.py`, and `src/belief.py`: intent, correction-aware state,
  slot extraction, confidence, clarification, and rationales;
- `src/retrieval.py`: optional BGE embeddings and rank fusion;
- `src/ranking.py`: coverage, need satisfaction, profile/quality priors, and retrieval safeguards;
- `src/reranker.py` and `src/ranking_features.py`: optional cross-encoder, Gemini, and linear LTR;
- `src/context_engine.py`: optional distilled context, profile persistence, and guidance learning;
- `src/trace.py` plus `app/`: structured traces and a local conversation-inspection UI.

The benchmark helper creates an isolated agent with persistence disabled so one evaluation session
cannot influence another.

## Development tools

- Python 3.10+
- pytest for unit, integration, evaluator, correction-ledger, and import-smoke coverage
- Ruff for static linting
- `compileall` for syntax/import compilation
- MyPy and Black as audit tools with remaining migration debt documented
- Git for version control
- Flask trace UI and structured JSONL traces for debugging

## APIs and models

- **SQLite FTS5 BM25:** required lexical retrieval path.
- **BAAI/bge-small-en-v1.5:** optional local query/product embeddings.
- **cross-encoder/ms-marco-MiniLM-L-6-v2:** optional local precision reranker.
- **Google Gemini via `google-genai`:** optional slot extraction, use-case inference, response
  generation, and listwise reranking; configured model ID is `gemini-flash-lite-latest`.
- **scikit-learn logistic regression:** experimental offline learning-to-rank training.

Gemini credentials are optional and environment-only. The verified public run used zero tokens.

## Libraries and frameworks

The baseline uses Python's standard library, including `sqlite3`. Optional paths use Flask, NumPy,
scikit-learn, sentence-transformers, google-genai, and python-dotenv. Dependency groups are declared
in `requirements-dev.txt` and `requirements-optional.txt`.

## Datasets and assets

The product catalog and sessions are derived from Amazon Reviews 2023, McAuley Lab, UCSD,
`Clothing_Shoes_and_Jewelry`. The project uses text and structured product metadata only. Direct user
identifiers, raw reviews, timestamps, product images, and the private evaluation set are not
included. Full attribution and use notes are in `DATA_ATTRIBUTION.md`.

The local UI uses repository-local inline code, emoji, and an inline SVG caret; it does not depend on stock
footage, music, product imagery, or third-party fonts.

## Key engineering decisions

- Keep the official scoring API small and stable while isolating all optional components behind
  availability checks.
- Build the catalog index once per agent process and keep downstream candidate operations bounded.
- Preserve corrections as explicit events so removed preferences do not silently re-enter the
  active query.
- Treat the public evaluator as a reproducibility guardrail while maintaining separate paraphrase
  stress tests because target-derived wording can overstate literal matching quality.
- Disable persistent context in benchmark runs to prevent cross-session contamination.
- Report current reproduced metrics, not stale or best-ever experiment numbers.

## Innovation and differentiation

The differentiation is the combination of benchmark-aware measurement discipline and a practical
offline fallback. The agent does not require an LLM to function, but it can opportunistically use
local semantic models or Gemini when they are available. Its state is corrective rather than purely
accumulative, and the trace UI makes retrieval, ranking, confidence, and reveal decisions visible
turn by turn.

This is not presented as a new foundation model or a production-scale recommender. The innovation is
in the orchestration of retrieval, state revision, ranking, clarification, and graceful degradation
for a constrained conversational-search setting.

## User impact and broader relevance

The approach can help people who know what outcome they want but not the catalog's terminology. It
also supports users who change their minds or decline to specify an attribute. Because recommendations
and questions are derived from explicit state and catalog evidence, the interaction is inspectable
and correctable rather than a one-shot opaque answer.

`[TEAM INPUT: add any validated user feedback or accessibility evidence. Do not add estimated reach,
conversion lift, or satisfaction metrics without a source.]`

## Feasibility beyond the hackathon

The core path runs locally with Python and SQLite, does not require a vector database, and makes no
mandatory network call. Optional models are modular and fail safe. These properties make a pilot
feasible on modest infrastructure.

Moving to production would still require a persistent search index, model-serving and cache
provenance, session eviction, concurrency controls, authentication, privacy/retention policies,
monitoring, load tests, and a defined deployment target. None of those are claimed as complete.

## Challenges encountered

- Separating genuine language understanding from the public simulator's target-derived wording.
- Balancing target recall, rank, and first-hit timing when revealing results early can freeze MRR.
- Handling preference corrections and intent overrides without reintroducing retired terms.
- Keeping optional dense, cross-encoder, and Gemini components from making startup fragile.
- Preventing durable profile state from contaminating independent evaluation sessions.
- Consolidating many measured experiment flags without rewriting the stabilized scoring path.

## Accomplishments

- A runnable standard-library fallback with no required API key.
- A stable evaluator-facing API and 138 passing automated tests.
- A full 200-session reproduced run: Hit Rate@10 `0.965`, MRR `0.852228`, MTTC `2.905`, technical
  score `0.900068`, and zero reported tokens.
- A separate 250-session local language-stress diagnostic completed at Hit Rate@10 `0.852`, MRR
  `0.4964`, MTTC `4.00`, and technical score `0.7148`; it is not presented as an official or
  real-user metric.
- Explicit state revision, override handling, retrieval/ranking separation, and failure fallbacks.
- A trace UI and structured debugging path for communicating how each result was produced.
- Complete reviewer, architecture, and submission documentation.

## Limitations

- Public evaluation is not representative of natural shopper language because of metadata overlap.
- Optional semantic/model paths were not exercised in the final deterministic validation.
- The largest modules remain complex and type checking is not yet clean.
- There is no production deployment, load test, formal latency result, or real-user study.
- The local trace UI is not authenticated or production-hardened.
- The team's original source code is MIT-licensed; datasets, downloaded model weights, APIs, and
  other third-party materials remain subject to their respective terms.

## Future improvements

- Run a human-written, target-independent language evaluation and usability study.
- Replace parallel phrase metadata lists with a typed state object and expand correction tests.
- Introduce incremental strict typing and split the turn orchestrator into smaller stages.
- Publish optional model/cache provenance plus cold/warm latency and memory measurements.
- Add browser/API tests, an accessibility audit, and production security/privacy controls.
- Select a deployment architecture only after expected traffic, latency, and data-retention needs are
  known.

## Links

- Repository: `[REPOSITORY_URL]`
- Setup guide: `[REPOSITORY_URL]#setup`
- Architecture: `[REPOSITORY_URL]/blob/main/architecture.md`
- Public YouTube demo: `[PUBLIC_YOUTUBE_URL]`

## Team contributions

- **Landon Zhao** — Led the modular conversational-search architecture, hybrid retrieval and ranking
  work, dialogue strategy, dynamic conversation policy, session context, intent-override handling,
  and ranking robustness evaluation.
- **Valerie Lim** — Developed and evaluated satisfaction-ranking and leak-free retrieval
  improvements, strengthened regression and experiment tooling, resolved integration issues, and
  led final repository stabilization, validation, documentation, and submission preparation.
- **Bryan Koh** — Implemented the correction-aware state ledger, evaluator isolation and retrieval
  optimizations, dual-track ranking and coverage mechanisms, structured clarification behavior, and
  contradiction/negation handling.
