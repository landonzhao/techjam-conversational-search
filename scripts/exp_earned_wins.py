"""Measure the two ranking signals from step 4 on public + paraphrase + hard tier.

  - price proximity: the disclosed budget is the target's own price, so a candidate priced near it
    is strong entity-resolution evidence (off by default, swept here).
  - structured coverage: normalized NeedModel constraints as a paraphrase-robust second track.

LLM is off, so gains are attributable to the deterministic signals alone. Each row flushes as it
completes, so progress is visible and nothing is lost to buffering.

Usage:  python -u scripts/exp_earned_wins.py
"""
from __future__ import annotations

import time

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from evaluator.robustness import run as robrun
from src.agent import Agent

CATALOG = "data/catalog.jsonl"


def main() -> None:
    pub = load_jsonl("data/public_set.jsonl")
    hard = [r for r in load_jsonl("data/synthetic_set.jsonl")
            if r.get("difficulty_bucket") == "hard"]
    cat_ids, cats, prods = catalog_index(CATALOG)

    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    agent = Agent(CATALOG)

    # (label, price_weight, structured_weight)
    rows = [
        ("baseline",          0.0, 0.0),
        ("price@2.0",         2.0, 0.0),
        ("price@3.0",         3.0, 0.0),
        ("price@2.0+struct.3", 2.0, 0.3),
    ]
    header = (f"{'config':>18} | {'PUBLIC':>7} {'hit':>6} {'mrr':>7} "
              f"| {'PARAPHR':>7} {'hit':>6} {'mrr':>6} | {'HARD':>7} {'hit':>6} {'mrr':>7}")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for label, pw, sw in rows:
        agent.USE_PRICE_PROXIMITY = pw > 0
        agent.PRICE_PROXIMITY_WEIGHT = pw
        agent.USE_STRUCTURED_COVERAGE = sw > 0
        agent.STRUCTURED_COVERAGE_WEIGHT = sw
        t0 = time.time()
        p = evaluate(agent, pub, cat_ids, cats, prods)
        r = robrun(agent, pub, cat_ids, cats, prods, "paraphrase")
        h = evaluate(agent, hard, cat_ids, cats, prods)
        print(f"{label:>18} | "
              f"{p['recommended_technical_score']:>7.4f} {p['hit_rate_at_10']:>6.3f} {p['mrr']:>7.4f} | "
              f"{r['score']:>7.4f} {r['hit@10']:>6.3f} {r['mrr']:>6.3f} | "
              f"{h['recommended_technical_score']:>7.4f} {h['hit_rate_at_10']:>6.3f} {h['mrr']:>7.4f} "
              f"  ({time.time()-t0:.0f}s)", flush=True)

    print("\nKeep price proximity if it holds/raises PUBLIC and helps HARD; keep structured if it "
          "raises PARAPHRASE without hurting PUBLIC.", flush=True)


if __name__ == "__main__":
    main()
