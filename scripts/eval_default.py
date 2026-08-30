"""Score the CURRENT DEFAULT config on public (leaderboard) + leak-free (honest), with the
per-scenario breakdown the organizer reports (Buying / Browsing / Intent Override / Boundary).

LLM is turned off: it is proven inert on the scored path, official scoring may disable network, and
this keeps the run fast and deterministic. The number therefore equals the default leaderboard score.

Usage:  python -u scripts/eval_default.py
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


def show(name: str, r: dict) -> None:
    print(f"\n{name}: TechnicalScore {r['recommended_technical_score']:.4f}  "
          f"(hit@10 {r['hit_rate_at_10']:.3f}  mrr {r['mrr']:.4f}  mttc {r['mttc']:.2f})", flush=True)
    for scen, m in sorted(r.get("scenario_metrics", {}).items()):
        print(f"    {scen:>16}: hit {m['hit_rate_at_10']:.3f}  mrr {m['mrr']:.4f}  mttc {m['mttc']:.2f}",
              flush=True)


def main() -> None:
    cat_ids, cats, prods = catalog_index(CATALOG)
    # current committed default — only LLM disabled (inert on score; network may be off in scoring)
    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    agent = new_isolated_agent(CATALOG)
    print("CURRENT DEFAULT CONFIG — official evaluator", flush=True)
    print(f"  retrieval_floor={agent.COVERAGE_RETRIEVAL_WEIGHT}  gate={agent.COVERAGE_INFORMATIVE_MIN}  "
          f"satisfaction={agent.USE_SATISFACTION_RANKER}", flush=True)

    t0 = time.time()
    show("PUBLIC (leaderboard)", evaluate(agent, load_jsonl("data/public_set.jsonl"),
                                          cat_ids, cats, prods))
    show("LEAK-FREE (honest)", evaluate(agent, load_jsonl("data/language_stress_set.jsonl"),
                                        cat_ids, cats, prods))
    print(f"\n({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
