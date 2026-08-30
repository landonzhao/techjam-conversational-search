# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## Our solution in one paragraph

A **deterministic, offline-first entity-resolution engine**. We read the evaluator end-to-end and
found the simulated customer is a *template engine, not a language model* — each session leaks at
most a few short strings lifted near-verbatim from the target product's own spec sheet. The winning
move is therefore entity resolution, not dialogue understanding, and the **core scored path runs
with zero external dependencies, no API key, and $0 cost**: BM25 + dense (BGE) hybrid retrieval →
verbatim-coverage reranking → adaptive reveal. On top of that core we add a **dense-retrieval track
and a structured-constraint signal to harden generalization** for the private set (where the
verbatim leak may not hold), and an **optional, token-metered LLM layer** for unseen language that
is *off by default and never on the critical path*. Every feature that ships on the scored path
earns its place with a measured ablation; everything else is explicitly labeled optional.

- **How it works internally:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Core vs optional components, and the measured status of every flag:** the flag ledger at the
  top of [`src/agent.py`](src/agent.py) and [`src/config.py`](src/config.py).

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/ARCHITECTURE.md              internal architecture — module map, component design, where to make changes
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  official submission entry point (imports from src/agent.py)
starter/agent_baseline.py         original weak BM25 starter (reference only)
evaluator/local_evaluator.py      public-set simulator and scorer (DO NOT MODIFY)
evaluator/robustness.py           paraphrase harness for held-out generalisation measurement
scripts/measure.py                run evaluator and compare configs
scripts/chat.py                   interactive REPL with :state / :reset
scripts/build_embeddings.py       precompute BGE embeddings for dense retrieval
tests/                            unit + integration tests (pytest)
src/                              implementation (see docs/ARCHITECTURE.md for module map)
```

## Running the Agent

```bash
# Run the official evaluator (full 200-session public set)
python scripts/measure.py

# Interactive chat REPL
python scripts/chat.py

# Run all tests
python -m pytest tests/ -v

# Paraphrase robustness harness (tests generalisation without the verbatim leak)
python -m evaluator.robustness
```

## Current Score

`TechnicalScore = 0.9297` (hit@10=0.995, MRR=0.887, MTTC=2.70, Efficiency=0.830)

Measured with the unmodified official evaluator on the 200 public sessions via `scripts/measure.py`,
on the **deterministic core path** — 0 tokens, no API key, $0. By scenario: buying `0.988`/MRR `0.884`,
browsing `1.000`/MRR `0.878`, intent_override `1.000`/MRR `0.904`, boundary `1.000`/MRR `0.933`.

See `docs/ARCHITECTURE.md` for the full architecture, component map, design decisions,
and the "Where Should I Make This Change?" reference table.

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
