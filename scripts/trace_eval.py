"""Run the OFFICIAL evaluator while capturing a full execution trace per session.

Scoring is authoritative: this reuses evaluator.local_evaluator.evaluate()
unchanged. A thin TracingAgent subclass only:
    * enables tracing (src/trace.py), and
    * correlates each evaluator session with its sample_id / ground truth so a
      trace can be tied back to the exact test case (and target rank per turn).

The evaluator drives one agent.reset() per sample, in dataset order, so a
sequential cursor over the same samples reliably identifies the active case.

Usage:
    python scripts/trace_eval.py --dataset data/test_suite/hard.jsonl \
        --trace-dir traces/hard --limit 50

Outputs:
    <trace-dir>/trace.jsonl   one JSON object per session (turns + reasoning)
    <trace-dir>/metrics.json  the evaluator's metric summary for this run

Then render a case:  python scripts/show_trace.py <trace-dir>/trace.jsonl --miss
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.eval_support import new_isolated_agent
from src.agent import Agent


class TracingAgent(Agent):
    """Agent that tags each session with its sample metadata for tracing.

    The official evaluate() loop calls reset() exactly once per sample, in
    order, so we advance a cursor over the bound samples on each reset().
    """

    def bind_samples(self, samples: list[dict]) -> None:
        self._trace_samples = list(samples)
        self._trace_cursor = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        samples = getattr(self, "_trace_samples", None)
        if samples and self._trace_cursor < len(samples):
            sample = samples[self._trace_cursor]
            self._trace_cursor += 1
            self._pending_meta = {
                "sample_id": sample.get("sample_id"),
                "scenario_type": sample.get("scenario_type"),
                "difficulty_bucket": sample.get("difficulty_bucket"),
                "category_bucket": sample.get("category_bucket"),
                "ground_truth": sample.get("ground_truth", {}).get("parent_asin"),
            }
        super().reset(session_id, user_profile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/synthetic_set.jsonl")
    parser.add_argument("--trace-dir", default="traces/run")
    parser.add_argument("--limit", type=int, default=0, help="use first N samples only")
    args = parser.parse_args()

    # Enable tracing for the agent constructed below (read in Agent.__init__).
    os.environ["AGENT_TRACE"] = "1"
    os.environ["AGENT_TRACE_DIR"] = args.trace_dir

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)

    agent = new_isolated_agent(args.catalog, agent_cls=TracingAgent)
    agent.bind_samples(samples)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    agent._tracer.close()

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    metrics = {k: v for k, v in result.items() if k != "sessions"}
    (trace_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"\ntraced {agent._tracer.sessions_written} sessions -> {trace_dir / 'trace.jsonl'}")


if __name__ == "__main__":
    main()
