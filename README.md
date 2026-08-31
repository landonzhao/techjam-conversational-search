# TechJam Conversational Search

An offline-first conversational shopping agent that asks focused follow-up questions and ranks a
hidden target product from a 50,000-item clothing, shoes, and jewelry catalog in at most ten turns.

The project addresses a practical search problem: shoppers often begin with incomplete language,
revise preferences, or browse without a precise query. The agent keeps correction-aware session
state, combines lexical retrieval with optional semantic retrieval, reranks candidates against the
active need, and decides whether to clarify or reveal recommendations.

## Main features

- Required `reset`/`respond` Agent API with isolated multi-turn session state.
- In-memory SQLite FTS5 BM25 retrieval over structured product metadata.
- Optional BGE dense retrieval and reciprocal-rank fusion with an offline BM25 fallback.
- Constraint ledger supporting additions, removals, clears, no-preference boundaries, and intent
  overrides.
- Need-satisfaction ranking, retrieval safeguards, confidence estimation, adaptive clarification,
  and adaptive result reveal.
- Optional local cross-encoder, Gemini-assisted language handling, and experimental linear LTR.
- Deterministic public evaluator, paraphrase robustness harness, unit/integration tests, tracing,
  an interactive CLI, and a local trace-inspection UI.

The complete implementation description and Mermaid diagrams are in
[`architecture.md`](architecture.md).

## Problem and approach

The challenge supplies an anonymized preference profile and a simulated shopper message. Each
turn, the agent may ask for one attribute, return up to ten catalog IDs, or do both. A session ends
when the target `parent_asin` appears in the scored top ten or after turn ten.

This solution uses an explicit staged pipeline:

```text
message + profile
  → intent and constraint-state update
  → BM25 retrieval (+ optional dense/expansion routes)
  → need-satisfaction ranking (+ optional rerankers)
  → belief, clarification, and reveal policy
  → contract-shaped response
```

The official public simulator can disclose phrases derived from the target catalog record. That
makes literal overlap more predictive than it would be for real shoppers. Public metrics are
therefore a competition guardrail, not a production-quality claim; the repository also includes
paraphrase-oriented stress data.

## Technology stack

| Area | Technology |
|---|---|
| Language/runtime | Python 3.10+ |
| Core retrieval | Python `sqlite3`, SQLite FTS5 BM25 |
| Optional semantic retrieval | NumPy, `sentence-transformers`, `BAAI/bge-small-en-v1.5` |
| Optional precision reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Optional hosted model | Google Gemini through `google-genai`; configured model is `gemini-flash-lite-latest` |
| Optional LTR tooling | NumPy and scikit-learn logistic regression |
| Local trace UI | Flask, HTML, CSS, and JavaScript |
| Validation | pytest, Ruff, `compileall`; MyPy/Black are audit-only pending migration |

The deterministic evaluator path uses only the Python standard library and needs no API key.
Optional packages and models fail safely to the deterministic path when unavailable.

## Repository map

| Path | Purpose |
|---|---|
| `starter/agent.py` | Official evaluator-facing entry point. |
| `src/` | Agent, state, retrieval, ranking, NLU, optional models, context, and tracing. |
| `evaluator/` | Local official-style evaluator and paraphrase robustness harness. |
| `tests/` | Unit, integration, correction-ledger, evaluator, and import-smoke tests. |
| `scripts/` | Evaluation, dataset-building, experiment, tracing, and demo utilities. |
| `app/` | Local Flask trace server and static inspector. |
| `data/` | Public and stress datasets; the decompressed catalog is local/ignored. |
| `prompts/` | Optional Gemini prompt templates. |
| `docs/` | Rules, decisions, experiments, audit notes, and submission materials. |

## Data, datasets, and assets

The catalog and sessions are derived from **Amazon Reviews 2023**, McAuley Lab, UCSD, category
`Clothing_Shoes_and_Jewelry`. See [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) before redistributing
or reusing the data.

Included tracked evaluation data:

- `data/public_set.jsonl`: 200 labeled development sessions.
- `data/language_stress_set.jsonl`: 250 paraphrase/language-stress sessions.
- `data/pillar_free.jsonl` and `data/pillar_moderate.jsonl`: 240 sessions each.
- `data/shadow/teaser.jsonl`: 10 small shadow-evaluation examples.

The 50,000-row `data/catalog.jsonl` is intentionally ignored because of its size. The expected
distribution method in the project materials is a GitHub Release archive. Before public submission,
replace the catalog-release placeholder in `docs/submission/submission-checklist.md` with the actual
public URL and published SHA-256 checksum.

The trace UI uses only inline CSS, JavaScript, emoji, and an inline SVG caret. It does not load
third-party images, fonts, music, or footage.

## Prerequisites

- Python 3.10 or later.
- A Python build whose SQLite includes FTS5. Standard current CPython builds normally do.
- Approximately 60 MB free for the decompressed catalog, plus optional model/cache space.
- Optional: network access for first-time Hugging Face model downloads or Gemini calls.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

The deterministic runtime has no third-party dependency, so this command is intentionally a no-op
apart from validating the manifest:

```bash
python -m pip install -r requirements.txt
```

Install test, lint, and local trace-UI tools:

```bash
python -m pip install -r requirements-dev.txt
```

Install optional semantic/LLM/LTR dependencies only when you intend to exercise those paths:

```bash
python -m pip install -r requirements-optional.txt
```

### Prepare the catalog

Download the public release asset to `data/catalog.jsonl.gz`, then:

```bash
gzip -t data/catalog.jsonl.gz
gzip -dk data/catalog.jsonl.gz
wc -l data/catalog.jsonl
```

The final command should print `50000`. Compare the archive's checksum with the checksum published
on the release page:

```bash
shasum -a 256 data/catalog.jsonl.gz
```

Do not commit the decompressed catalog, private evaluation data, generated caches, or credentials.

## Environment variables

No environment variable is required for the baseline. Copy the example only for optional features:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | No | Primary Gemini API credential. |
| `GOOGLE_API_KEY` | No | Alternative primary credential name. |
| `GEMINI_API_KEY_2` ... `_31` | No | Optional additional keys used for quota rotation. |
| `AGENT_TRACE` | No | Set to `1` to write structured turn traces. |
| `AGENT_TRACE_DIR` | No | Trace output directory; defaults to `traces`. |

Never put real values in `.env.example`, source files, screenshots, traces, or submission prose.

## Run the project

### Official-style local evaluation

```bash
python3 scripts/measure.py
```

The verified result at commit `d2ef469`, with optional model assets unavailable, is:

```text
sample_count=200
hit_rate_at_10=0.965
mrr=0.852228
mttc=2.905
efficiency=0.8095
recommended_technical_score=0.900068
reported_token_usage=0
```

The separate, synthetic `data/language_stress_set.jsonl` diagnostic completed with Hit Rate@10
`0.852`, MRR `0.4964`, MTTC `4.00`, and technical score `0.7148` via
`python3 -u scripts/eval_default.py`. This is a local paraphrase-stress result, not an official
leaderboard or real-user metric.

You can also use the evaluator entry point, which writes ignored `results.json`:

```bash
python3 -m evaluator.local_evaluator
```

### Interactive development demo

```bash
python3 scripts/chat.py --top-k 5
```

Enter shopper messages at the prompt. Use `:state`, `:reset`, and `:quit` to inspect or control the
session.

### Local trace UI

```bash
python3 app/trace_server.py
```

Open `http://127.0.0.1:5001`, select a dataset/sample, and run the simulation. This Flask server is
developer tooling, not a production service; do not expose it publicly.

### Paraphrase robustness evaluation

The full four-configuration harness is substantially slower than the public evaluator:

```bash
python3 -m evaluator.robustness
```

Use a bounded subset while iterating:

```bash
python3 -m evaluator.robustness --limit 40
```

### Optional dense embeddings

This downloads a Hugging Face model on first use and writes ignored files under `cache/`:

```bash
python3 scripts/build_embeddings.py
```

The agent automatically uses the cache on later runs. If the files or package are absent, it uses
BM25 instead.

## Reproduce the representative workflow

1. Prepare `data/catalog.jsonl`.
2. Run `python3 scripts/measure.py` to verify all 200 public sessions.
3. Run `python3 scripts/chat.py --top-k 5` for a manual conversation, or use the trace UI.
4. For the first public sample, the evaluator begins with:

```text
I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.
```

On the verified deterministic run, turn one asks for `other` and holds back one recommendation.
After the simulator reveals the next feature phrase, turn two returns target `B09PYB7B6Z` at rank 1
with zero reported model tokens. Exact customer-facing wording can include product-derived
rationales, but every response preserves this structure:

```python
{
    "message": "<customer-facing question or rationale>",
    "ask_attribute": "other",
    "recommendations": [{"parent_asin": "B09PYB7B6Z"}],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
}
```

## Validation commands

Run from the activated environment:

```bash
# Lint
ruff check src starter evaluator app scripts tests

# Unit/integration tests
python -m pytest tests/ -q

# Syntax/import compilation (the repository has no separate production build step)
python -m compileall -q src starter evaluator app scripts tests

# Full public evaluation
python3 scripts/measure.py

# Current public + full language-stress diagnostic
python3 -u scripts/eval_default.py

# Local application startup
python3 app/trace_server.py --host 127.0.0.1 --port 5001
```

Type-check and formatting audit commands are valid but currently expose pre-existing repository
debt; they are not green gates and were not weakened during cleanup:

```bash
mypy --explicit-package-bases --ignore-missing-imports src starter evaluator app scripts tests
black --check src starter evaluator app scripts tests
```

See `docs/submission/submission-checklist.md` for recorded results and exact known failures.

## Troubleshooting

### `No module named pytest`

The shell may be using a different Python than the one that owns `pytest`. Activate `.venv` and run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

### `data/catalog.jsonl` is missing

Download/decompress the catalog release asset as described above. Confirm the file has 50,000
records and that you are running commands from the repository root.

### Dense retrieval or cross-encoding does not activate

Install `requirements-optional.txt`, build the embedding cache, and allow the relevant model to
download once. The baseline intentionally continues with BM25 when optional assets are unavailable.

### Gemini remains unavailable

Install the optional requirements and set a valid environment variable in ignored `.env`. Gemini is
not needed for scoring, and failed/missing API access deliberately returns the deterministic result.

### SQLite reports `no such module: fts5`

Use a CPython/SQLite build with FTS5 enabled. You can check support with:

```bash
python3 -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('create virtual table t using fts5(x)'); print('FTS5 OK')"
```

### Scores differ from the verified baseline

Confirm the current commit, catalog checksum, Python command, absence/presence of optional model
caches, and that benchmark evaluation uses `scripts/measure.py`, which isolates persistent DCP
state. Do not compare regenerated datasets with the frozen tracked sets.

## Known limitations

- Public simulator wording overlaps target metadata and can inflate lexical methods.
- Optional ML/model paths are not required and were not exercised in the final deterministic score.
- Large orchestration and understanding modules remain; a late structural rewrite would be risky.
- MyPy and Black audits are not yet clean repository-wide.
- The local Flask UI is not production-hardened or authenticated.
- No load, concurrency, memory, or formal latency benchmark has been recorded.
- Durable profiles are local best-effort JSON storage and need a privacy/retention design for real
  users.
- No production deployment target is defined.

## Improvements with more time

- Replace parallel phrase metadata lists with one typed record and extend override tests.
- Make type checking incremental and mandatory by module.
- Split the orchestrator into explicit understand/retrieve/rank/respond stages.
- Publish verified optional-model/cache provenance and cold/warm resource measurements.
- Add browser/API tests and an accessibility audit for the trace UI.
- Evaluate with human-written requests that are independent of target product text.
- Add a deployment target, observability, session eviction, and concurrency controls if the project
  becomes a service.

## Team contributions

`[TEAM INPUT REQUIRED: add each member's name and specific product, engineering, data, evaluation,
design, documentation, and presentation contributions before submission.]`

Git history contains contributor identities, but this README does not infer team roles from commit
metadata.

## License

No repository license has been selected. All rights remain with their respective owners unless a
license is added. Before public submission, the team must choose a source-code license and verify
that dataset, model, API, and asset terms are compatible. Amazon Reviews 2023 attribution and data
use notes are in [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md).
