"""Sweep COVERAGE_RETRIEVAL_WEIGHT on public (guardrail) + leak-free (target).

The deterministic paraphrase lever: when verbatim coverage is uninformative (paraphrased words →
all coverage scores ~0), the reranker currently collapses to a popularity tie-break, discarding the
SEMANTIC order dense retrieval already produced. COVERAGE_RETRIEVAL_WEIGHT RRF-fuses the retrieval
(dense+BM25) order back in as a floor, so the semantic ranking carries the paraphrase cases — no LLM.

Goal: a weight that LIFTS leak-free without tanking public (where verbatim coverage rightly wins).
LLM off, deterministic. Rows flush as they complete.

Usage:  python -u scripts/exp_retrieval_weight.py
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.eval_support import new_isolated_agent
from src.agent import Agent

CATALOG = "data/catalog.jsonl"
WEIGHTS = [0.0, 0.5, 1.0, 2.0]


def main() -> None:
    pub = load_jsonl("data/public_set.jsonl")
    leak = load_jsonl("data/language_stress_set.jsonl")
    cat_ids, cats, prods = catalog_index(CATALOG)

    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    agent = new_isolated_agent(CATALOG)

    header = f"{'w_ret':>6} | {'PUBLIC':>7} {'hit':>6} {'mrr':>7} | {'LEAK-FREE':>9} {'hit':>6} {'mrr':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for w in WEIGHTS:
        agent.COVERAGE_RETRIEVAL_WEIGHT = w
        t0 = time.time()
        p = evaluate(agent, pub, cat_ids, cats, prods)
        lk = evaluate(agent, leak, cat_ids, cats, prods)
        print(f"{w:>6.1f} | "
              f"{p['recommended_technical_score']:>7.4f} {p['hit_rate_at_10']:>6.3f} {p['mrr']:>7.4f} | "
              f"{lk['recommended_technical_score']:>9.4f} {lk['hit_rate_at_10']:>6.3f} {lk['mrr']:>7.4f} "
              f"  ({time.time()-t0:.0f}s)", flush=True)

    print("\nKeep the weight that raises LEAK-FREE while holding PUBLIC near 0.9297.", flush=True)


if __name__ == "__main__":
    main()
