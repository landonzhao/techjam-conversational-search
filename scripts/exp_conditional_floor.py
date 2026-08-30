"""Validate the conditional (informativeness-gated) retrieval floor — P0.1.

Flat COVERAGE_RETRIEVAL_WEIGHT trades public for leak-free (1.0 -> 0.9171/0.385, 2.0 -> 0.8923/0.605).
The gate applies the floor ONLY on paraphrased turns (top per-phrase coverage < COVERAGE_INFORMATIVE_MIN),
so verbatim turns keep the pure-coverage sort. Hypothesis: the gate lets the paraphrase branch run at a
HIGH weight to reach the flat-2.0 leak-free level while public stays ~0.9297 — best of both columns.

Exit (docs/STRENGTHENING_PLAN.md P0.1): public >= 0.928 AND leak-free >= 0.50.

Usage:  python -u scripts/exp_conditional_floor.py
"""
from __future__ import annotations

import time

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src.agent import Agent

CATALOG = "data/catalog.jsonl"
PUBLIC_GUARD_N = 100   # public is the set we PASS; sample it as a cheap guardrail, don't run all 250
# (label, informative_min, retrieval_weight) — the decisive configs only
CONFIGS = [
    ("flat w_ret=1.0 (gate off)", 0.0, 1.0),   # control: the currently-wired flat baseline
    ("gated min=1.0 w=2.0",       1.0, 2.0),    # the bet: push paraphrase branch, gate protects public
    ("gated min=1.0 w=3.0",       1.0, 3.0),
]


def main() -> None:
    pub = load_jsonl("data/public_set.jsonl")[:PUBLIC_GUARD_N]  # guardrail sample only
    leak = load_jsonl("data/language_stress_set.jsonl")         # the failing set — run in full
    cat_ids, cats, prods = catalog_index(CATALOG)

    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    agent = Agent(CATALOG)

    header = (f"{'config':>26} | {'PUB-'+str(PUBLIC_GUARD_N):>7} {'hit':>6} {'mrr':>7} | "
              f"{'LEAK-FREE':>9} {'hit':>6} {'mrr':>7}")
    print(f"conditional retrieval floor — public guardrail={PUBLIC_GUARD_N} (full-set ref 0.9297), "
          f"leak-free=250 (the failing set)", flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for label, imin, w in CONFIGS:
        agent.COVERAGE_INFORMATIVE_MIN = imin
        agent.COVERAGE_RETRIEVAL_WEIGHT = w
        t0 = time.time()
        p = evaluate(agent, pub, cat_ids, cats, prods)
        lk = evaluate(agent, leak, cat_ids, cats, prods)
        print(f"{label:>26} | "
              f"{p['recommended_technical_score']:>7.4f} {p['hit_rate_at_10']:>6.3f} {p['mrr']:>7.4f} | "
              f"{lk['recommended_technical_score']:>9.4f} {lk['hit_rate_at_10']:>6.3f} {lk['mrr']:>7.4f} "
              f"  ({time.time()-t0:.0f}s)", flush=True)

    print("\nKeep the config that meets public >= 0.928 AND leak-free >= 0.50.", flush=True)


if __name__ == "__main__":
    main()
