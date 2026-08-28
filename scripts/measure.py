"""Measure any Agent class against the official evaluator's evaluate() function.

Usage:
    python scripts/measure.py                   # full 200 sessions
    python scripts/measure.py --limit 40        # quick dev subset
    python scripts/measure.py --agent src.agent
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="src.agent", help="module exporting Agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="use first N sessions only")
    args = parser.parse_args()

    Agent = importlib.import_module(args.agent).Agent
    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    print(json.dumps({k: v for k, v in result.items() if k != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
