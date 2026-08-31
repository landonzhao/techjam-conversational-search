# TechJam Conversational Search Architecture

This document describes the implementation at commit `d2ef469` as verified on 31 August 2026.
Configuration flags describe what the code attempts to enable; optional components only execute
when their local package, model, cache, and (for Gemini) credentials are available.

## Purpose and problem statement

The project is a multi-turn shopping agent for a frozen 50,000-product clothing, shoes, and
jewelry catalog. A shopper provides an anonymized aggregate profile and progressively reveals a
need. The agent has at most ten turns to ask useful follow-up questions and return the hidden target
product as early and as highly ranked as possible.

The local evaluator scores exact `parent_asin` matches using Hit Rate@10, mean reciprocal rank
(MRR), and mean turns to conversion (MTTC). The public simulator derives some shopper constraints
from target-product metadata, so literal catalog overlap is unusually predictive. The repository
therefore keeps the official public evaluation as a guardrail and also includes paraphrase-oriented
stress sets to expose weaker real-language generalization. Neither local evaluation is evidence of
production performance.

## Solution overview

The required `Agent` API is implemented as an offline-first pipeline:

1. Maintain isolated state for each conversation.
2. detect buying, browsing, boundary, and override cues;
3. extract structured preferences and catalog-backed constraint phrases;
4. retrieve a candidate pool with SQLite FTS5 BM25, with optional dense retrieval;
5. add low-weight synonym, use-case, and profile expansion signals;
6. rank candidates using lexical/semantic need satisfaction, optional local reranking, and
   correction-aware state;
7. estimate confidence, choose a clarification action, and adapt how many results are revealed;
8. return the contract-shaped response and token usage.

The default evaluator path has a standard-library fallback and needs no API key. Dense retrieval,
cross-encoding, Gemini inference, the trace UI, and learning-to-rank tooling are optional.

## System context

```mermaid
flowchart LR
    Shopper[Shopper or local evaluator] -->|reset/respond| Agent[starter.agent.Agent]
    Agent --> State[Conversation state and preference ledger]
    Agent --> Retrieval[Candidate retrieval]
    Retrieval --> Catalog[(50k-product JSONL catalog)]
    Retrieval --> BM25[SQLite FTS5 BM25]
    Retrieval -. optional .-> Dense[BGE embeddings and encoder]
    Agent --> Ranking[Need-satisfaction ranking]
    Ranking -. optional .-> CE[MS MARCO cross-encoder]
    Ranking -. optional .-> Gemini[Gemini API]
    Agent --> Dialogue[Belief, clarification, reveal policy]
    Agent -->|message, ask_attribute, recommendations, usage| Shopper
    Evaluator[Official-style evaluator] --> Agent
    Evaluator --> Metrics[Hit@10, MRR, MTTC, technical score]
    TraceUI[Flask trace UI] --> Evaluator
```

## Major components

| Area | Implementation | Responsibility |
|---|---|---|
| Public entry point | `starter/agent.py` | Re-exports `src.agent.Agent` without changing the required interface. |
| Orchestration | `src/agent.py` | Builds components, owns feature flags, and executes each turn. |
| Configuration | `src/config.py` | Central weights, thresholds, cache paths, model IDs, and default feature settings. |
| Catalog and lexical retrieval | `src/catalog.py` | Loads JSONL, normalizes text, and builds an in-memory SQLite FTS5 index. |
| Dense retrieval and fusion | `src/retrieval.py` | Loads BGE embeddings, encodes queries, and performs reciprocal-rank or convex fusion. |
| Dialogue state | `src/dialogue.py` | Holds per-session state, intent routing, constraint-marker parsing, phase transitions, and response templates. |
| Natural-language understanding | `src/understanding.py` | Constraint ledger, slot filling, category resolution, vocabulary, expansions, and use-case inference. |
| Belief and explanation | `src/belief.py` | Confidence/convergence logic, question selection, filters, and rationales. |
| Ranking | `src/ranking.py` | Coverage, need satisfaction, popularity controls, retrieval guard, personalization, and diversity. |
| Optional rerankers | `src/reranker.py`, `src/ranking_features.py` | Local cross-encoder, Gemini listwise reranker, and linear LTR scoring. |
| Optional LLM layer | `src/llm_inference.py`, `src/keys.py`, `prompts/` | Gemini slot extraction, use-case inference, response generation, key rotation, caching, and token accounting. |
| Context/personalization | `src/context_engine.py` | Distilled context, durable preference profiles, orchestration plan, and clarification guidance. |
| Tracing | `src/trace.py` | Opt-in structured turn traces controlled by environment variables. |
| Evaluation | `evaluator/` | Official-style simulator/scorer and a paraphrase robustness harness. |
| Developer UI | `app/trace_server.py`, `app/static/index.html` | Local Flask API and browser-based conversation/trace inspector. |

## Repository structure

```text
.
├── app/                     # local trace server and static inspector
├── data/                    # public/stress datasets; catalog is downloaded separately
├── docs/                    # decisions, experiments, rules, schemas, and supporting design notes
├── docs/submission/         # public-submission drafts and readiness checklists
├── evaluator/               # deterministic local evaluation and robustness harness
├── prompts/                 # optional Gemini prompt templates
├── scripts/                 # evaluation, data-building, experiment, tracing, and demo tools
├── src/                     # first-party agent implementation
├── starter/                 # required evaluator-facing import path
├── tests/                   # pytest unit, integration, regression, and import-smoke tests
├── architecture.md          # this implementation overview
└── README.md                # reviewer setup and reproduction guide
```

Generated and local runtime artifacts are intentionally excluded from Git: the decompressed
catalog, caches, embeddings, traces, evaluator results, Python caches, and `.env`.

## End-to-end turn flow

```mermaid
sequenceDiagram
    participant E as Evaluator or caller
    participant A as Agent
    participant S as ConversationState
    participant R as Retrieval
    participant K as Ranking
    participant D as Dialogue policy

    E->>A: reset(session_id, user_profile)
    A->>S: create isolated state
    E->>A: respond(session_id, message, turn, top_k)
    A->>S: append message and revise constraint ledger
    A->>A: route intent and detect overrides/boundaries
    A->>R: retrieve active query, pool size
    R->>R: BM25; optional dense; optional expansion; fuse
    R-->>A: ordered parent_asin pool
    A->>K: satisfaction/coverage ranking and optional rerankers
    K-->>A: reordered pool and score map
    A->>D: update belief and choose ask/reveal action
    D-->>A: message, ask_attribute, reveal count
    A-->>E: recommendations and usage counters
```

In the local evaluator, a successful session stops when the target appears in the first ten valid,
unique recommendations. Intent-override sessions cannot score before the override message is sent.
The benchmark helper in `scripts/eval_support.py` disables durable DCP state and uses a temporary
directory so sessions remain isolated and reproducible.

## Frontend and backend

The judged artifact is the Python `Agent`, not a deployed web product. The repository nevertheless
contains a developer-facing trace UI:

- `app/trace_server.py` serves a single static page and JSON endpoints with Flask.
- `GET /api/datasets` lists local JSONL datasets.
- `GET /api/samples?path=...` lists up to 500 samples from a repository-confined path.
- `GET /api/traces` and `GET /api/trace?path=...` expose saved local traces.
- `POST /api/simulate` evaluates one selected sample and returns its in-memory trace.
- `app/static/index.html` renders the shopper/assistant exchange, recommendations, and stage data.

The server binds to `127.0.0.1:5001` by default. It has no authentication, multi-user isolation,
or production WSGI configuration and should remain local development tooling.

## Retrieval, NLP, ranking, and model components

### Catalog and BM25

`Catalog` reads the entire catalog into a `dict[parent_asin, product]` and builds an in-memory
SQLite FTS5 virtual table over title, categories, features, details, store, and description.
Field weights and the maximum query-term count are defined in `src/config.py`. This path is the
reliable fallback when every optional dependency is absent.

### Dense retrieval

`VectorRetriever` expects `cache/embeddings.npy` and `cache/asins.json`, produced by
`scripts/build_embeddings.py` with `BAAI/bge-small-en-v1.5`. It uses normalized embeddings and dot
products as cosine similarity. If the package, model, or cache is unavailable, construction fails
inside a guarded block and the agent continues with BM25. No embedding cache is committed.

### Understanding and state

The state layer combines:

- raw message history for audit and selected compatibility paths;
- a structured constraint ledger with `SET`, `ADD`, `REMOVE`, `CLEAR`, and `NO_PREFERENCE` events;
- regex-based slot/category extraction;
- active and excluded preference projections;
- intent score, phase, belief, boundary attributes, and override metadata;
- optional persistent profile and guidance objects.

The most recent commit adds a parallel weight list for simulator-derived constraint phrases.
When an explicit override introduces a new phrase, earlier phrases are multiplied by
`OVERRIDE_PHRASE_DEMOTE` (`0.3`) rather than discarded. This behavior is part of the current
ranking implementation. The full suite still passes after the addition, but there is no dedicated
unit test for phrase-weight invalidation.

### Ranking and dialogue policy

The default satisfaction scorer combines lexical phrase coverage with semantic similarity when a
vector model is available, then applies bounded popularity/quality priors and retrieval safeguards.
The cross-encoder flag is enabled in configuration but the component silently becomes unavailable
when `sentence-transformers` or the model cannot load. Gemini reranking and LTR are disabled by
default. Belief and convergence state drive clarification wording and adaptive reveal; the latter
may intentionally show fewer than `top_k` items while confidence is low.

### Optional Gemini layer

All Gemini calls pass through `GeminiClientPool`. It discovers `GEMINI_API_KEY`,
`GOOGLE_API_KEY`, and sequential `GEMINI_API_KEY_2` ... `GEMINI_API_KEY_31` values. It rotates keys
only after quota/rate-limit errors and records process-wide prompt/completion token totals.
Missing SDKs, missing keys, invalid responses, and request errors fall back to deterministic logic.

## APIs, models, libraries, tools, datasets, and assets

| Kind | Name | Use | Required for baseline? |
|---|---|---|---|
| Standard library | Python 3.10+, SQLite/FTS5, JSON, pathlib, dataclasses | Core agent and evaluator | Yes |
| Test/lint | pytest, Ruff | Automated validation | Development only |
| Local UI | Flask | Trace server | No |
| Numeric/ML | NumPy, scikit-learn | Embeddings and LTR training | No |
| Embedding/reranking | sentence-transformers | BGE retriever and MS MARCO cross-encoder | No |
| Hosted model API | `google-genai`, Gemini `gemini-flash-lite-latest` | Optional slots, use-case, response, reranking | No |
| Environment loading | python-dotenv | Optional `.env` loading | No |
| Dataset | Amazon Reviews 2023, `Clothing_Shoes_and_Jewelry` | Product metadata source | Catalog required |
| Public sessions | `data/public_set.jsonl` | 200 labeled local sessions | Yes for public evaluation |
| Stress sessions | `language_stress_set.jsonl`, `pillar_*.jsonl`, `shadow/teaser.jsonl` | Local robustness experiments | No |
| UI assets | Repository-local inline CSS, JavaScript, emoji, and an inline SVG caret | Local trace UI | No external asset fetch |

Data attribution and use constraints are documented in `DATA_ATTRIBUTION.md`. The repository does
not include product images, review text, raw user identities, or the private evaluation set.

## Configuration and environment variables

Behavioral configuration is code-based in `src/config.py` and the `Agent` flag ledger. Important
runtime paths include the catalog, embedding cache, LTR model, profile store, guidance store, LLM
caches, and trace output directory.

Environment variables:

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | No | Primary optional Gemini credential. |
| `GOOGLE_API_KEY` | No | Fallback name for the primary Gemini credential. |
| `GEMINI_API_KEY_2` ... `_31` | No | Additional keys for quota rotation. |
| `AGENT_TRACE` | No | Truthy value enables structured tracing. |
| `AGENT_TRACE_DIR` | No | Trace directory; defaults to `traces`. |

Never commit real values. `.env` and `cache/` are ignored. The deterministic evaluator runs with no
environment variables.

## Key technical decisions

- **Offline-first execution.** A standard-library path remains runnable without network, model
  downloads, or credentials, which matches final-scoring constraints.
- **Retrieve, then rank.** Broad candidate recall is separated from precision ordering so each can
  be measured and changed independently.
- **Hybrid retrieval is opportunistic.** Dense retrieval improves semantic recall when its local
  assets exist but is not allowed to make startup fragile.
- **Explicit state and corrections.** The constraint ledger records both active preferences and
  corrections instead of treating the raw transcript as an always-valid query.
- **Evidence-aware ranking.** Lexical coverage remains valuable for the benchmark distribution;
  semantic and retrieval safeguards limit over-reliance on exact wording.
- **Optional expensive layers fail closed.** Cross-encoder, Gemini, LTR, and persistent context are
  isolated behind availability checks and flags.
- **Benchmark isolation.** Evaluation disables persistent profiles and fixes the pool size through
  `new_isolated_agent` so local state cannot contaminate scores.
- **Measured claims are separated from design intent.** Experiment and decision documents preserve
  prior ablation context, while the README reports only the score reproduced in this cleanup.

## Error handling and fallback behavior

- `respond` raises a clear error if `reset` was not called; the evaluator converts agent exceptions
  or invalid responses into misses rather than crashing the complete run.
- Missing dense caches/models fall back to BM25.
- Cross-encoder prediction failures preserve the incoming candidate order.
- Gemini initialization/calls/parsing return deterministic fallbacks; token usage remains zero when
  no successful API call occurs.
- Profile, guidance, and LLM-cache reads treat corrupt/missing JSON as empty state; writes are
  best-effort so persistence cannot stop a scored run.
- Trace output is disabled by default and does not affect core execution.
- The trace server confines requested paths to the repository root.

Broad exception handling is deliberate around optional infrastructure, but it can make diagnostics
quiet. The tracer and targeted scripts are the current observability mechanisms.

## Security and privacy

- Credentials are environment-only; `.env` is ignored and `.env.example` contains placeholders.
- The checked source and documentation contain no detected credential-shaped values or private-key
  blocks as of the verification date.
- User profiles contain aggregate preference/rating summaries. Persistent profile identifiers are
  derived from a SHA-1 hash of supplied profile content when no user ID exists. This is an opaque
  lookup key, not anonymization against re-identification.
- Local caches may contain derived preferences and model outputs; operators should treat `cache/`
  and `traces/` as potentially sensitive runtime data.
- The developer Flask server is not hardened for public exposure.
- The source dataset has its own terms; redistribution must follow `DATA_ATTRIBUTION.md` and the
  source provider's applicable conditions.

## Performance and scalability

Startup reads 50,000 JSONL records, stores product dictionaries in memory, and builds an in-memory
FTS5 index. Optional dense mode additionally memory-maps/loads an embedding matrix and initializes
an encoder; the cross-encoder may also load a separate model. Per-turn work is bounded by configured
candidate-pool and rerank depths, while state is scoped by session ID.

The architecture is suitable for a single-process hackathon evaluator and local demo. It has no
measured concurrency, memory, latency, or load-test results in the repository. The global cached
agent in the Flask tool is intentionally serialized with `threaded=False`. Scaling beyond this
shape would require persistent indexes, controlled model serving, explicit concurrency safety,
bounded session eviction, and production telemetry.

## Runtime and deployment architecture

There is no production deployment manifest, container, cloud resource, or hosted endpoint in this
repository. Supported runtime modes are:

1. **Evaluator/library mode:** Python imports `starter.agent.Agent` and calls `reset`/`respond`.
2. **CLI evaluation mode:** `scripts/measure.py` or `evaluator.local_evaluator` loads data and scores
   sessions in one process.
3. **Interactive CLI mode:** `scripts/chat.py` accepts manual shopper messages.
4. **Local trace UI mode:** Flask serves the static inspector and runs selected samples serially.

Production deployment is therefore unverified and intentionally not claimed.

## Known limitations and technical debt

- `src/agent.py`, `src/ranking.py`, and `src/understanding.py` remain large and have many feature
  branches. Splitting them would be a meaningful future refactor but is too risky for final cleanup.
- Type checking is not yet clean. An exploratory MyPy run reports many pre-existing annotation and
  dynamic-monkeypatch issues; there is no committed strict type-check configuration.
- No repository-wide formatter was configured before cleanup. Ruff lint is now clean, but applying
  Black to the entire historical codebase would create a large non-functional diff.
- Optional ML packages/models and embedding/LTR caches are not part of the baseline repository and
  were not exercised in the final deterministic evaluation.
- The phrase-demotion commit stores phrases, turns, and weights in parallel lists. The existing
  historical-phrase invalidation helper updates phrases/turns but not weights; if that optional
  correction path is enabled, weighting may fall back to equal weights because list lengths differ.
- Public simulator constraints can overlap target metadata verbatim, so the reproduced public score
  should not be presented as unbiased real-shopper performance.
- Clarification and adaptive reveal are optimized partly around evaluator behavior and still need
  user testing.
- Durable profile storage is local JSON with best-effort writes, not a transactional data store.
- The trace UI has no authentication, accessibility audit, or browser-test suite.
- A prompt file for query rewriting exists but has no active consumer.
- No license has been selected for the repository itself.

## Practical future improvements

Without changing the current submission behavior, the next engineering steps should be:

1. introduce typed candidate/state objects and make strict type checking incremental by module;
2. replace parallel phrase metadata lists with one typed record;
3. split orchestration into explicit understand/retrieve/rank/respond stages;
4. add unit tests for phrase-weight invalidation, optional-component fallbacks, Flask errors, and
   profile corruption/concurrency;
5. pin and verify an optional ML environment, publish model/cache provenance, and record cold/warm
   latency and memory measurements;
6. add a container or deployment manifest only after a production target is chosen;
7. run a human-language evaluation independent of target-derived simulator phrasing;
8. complete a privacy review and retention policy before storing real-user profiles or traces;
9. select and add an explicit repository license with dataset/model license compatibility review.

## Verified baseline

On the current checkout, using Python 3.13.5 for the deterministic application command and the
available Conda pytest runner:

- `pytest tests/ -q`: 138 passed.
- `python3 -m compileall -q src starter evaluator app scripts tests`: passed.
- `ruff check src starter evaluator app scripts tests`: passed after mechanical cleanup.
- `python3 scripts/measure.py`: 200 sessions, Hit Rate@10 `0.965`, MRR `0.852228`, MTTC `2.905`,
  efficiency `0.8095`, technical score `0.900068`, reported token use `0`.
- `python3 -u scripts/eval_default.py`: the same public result plus the full 250-session local
  language-stress diagnostic at Hit Rate@10 `0.852`, MRR `0.4964`, MTTC `4.00`, technical score
  `0.7148`; this diagnostic is not an official or real-user metric.

See `docs/submission/submission-checklist.md` for the complete validation record and unresolved
environment/tooling caveats.
