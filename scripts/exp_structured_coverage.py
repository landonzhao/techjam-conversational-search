"""Initiative A measurement — structured constraint coverage, swept on three sets.

Structured coverage is a second ranking track scoring candidates by satisfaction of NORMALIZED
NeedModel constraints (regex + LLM slots), RRF-fused into the verbatim coverage order. The thesis:
it holds the public (verbatim-leak) score while RISING on paraphrase (where verbatim collapses),
because it matches normalized values rather than leaked tokens.

This first pass runs LLM OFF, so any gain is attributable to structured coverage over the
DETERMINISTIC regex slots alone. A follow-up can add LLM slots on top.

Guardrail to beat (current deterministic baseline, w=0.0):
  public >= ~0.915 must hold · paraphrase should rise from ~0.604 · hard tier must not regress.

Usage:  python scripts/exp_structured_coverage.py
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from evaluator.robustness import run as robrun
from scripts.eval_support import new_isolated_agent
from src.agent import Agent

CATALOG = "data/catalog.jsonl"
PUBLIC = "data/public_set.jsonl"
SYNTH = "data/synthetic_set.jsonl"
WEIGHTS = [0.0, 0.3, 0.6, 1.0]


def run() -> None:
    pub = load_jsonl(PUBLIC)
    hard = [r for r in load_jsonl(SYNTH) if r.get("difficulty_bucket") == "hard"]
    cat_ids, cats, prods = catalog_index(CATALOG)

    # LLM fully off: isolate structured coverage over deterministic regex slots.
    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    agent = new_isolated_agent(CATALOG)

    header = (f"{'w_struct':>8} | {'PUBLIC':>7} {'hit':>6} {'mrr':>7} "
              f"| {'PARAPHR':>7} {'hit':>6} {'mrr':>6} | {'HARD':>7} {'hit':>6} {'mrr':>7}")
    print(header)
    print("-" * len(header))
    for w in WEIGHTS:
        agent.USE_STRUCTURED_COVERAGE = w > 0
        agent.STRUCTURED_COVERAGE_WEIGHT = w
        t0 = time.time()
        p = evaluate(agent, pub, cat_ids, cats, prods)
        r = robrun(agent, pub, cat_ids, cats, prods, "paraphrase")
        h = evaluate(agent, hard, cat_ids, cats, prods)
        print(f"{w:>8.1f} | "
              f"{p['recommended_technical_score']:>7.4f} {p['hit_rate_at_10']:>6.3f} {p['mrr']:>7.4f} | "
              f"{r['score']:>7.4f} {r['hit@10']:>6.3f} {r['mrr']:>6.3f} | "
              f"{h['recommended_technical_score']:>7.4f} {h['hit_rate_at_10']:>6.3f} {h['mrr']:>7.4f} "
              f"  ({time.time()-t0:.0f}s)")

    print("\nKeep only if a single w holds PUBLIC (>= ~0.915) AND raises PARAPHRASE without")
    print("regressing HARD. Otherwise revert to flag-off (kill-criterion).")


if __name__ == "__main__":
    run()
