"""Fix #1 validation — should the NeedSatisfactionScorer become the default ranker?

Compares the current default (coverage + flat retrieval floor) against the satisfaction ranker at a
few adaptive-popularity weights, across the leak spectrum. Decision rule: adopt the satisfaction
config that beats the default on the honest sets (pillar_moderate / pillar_free) while holding public
within ~0.015 of the current 0.9172 leaderboard number.

public is the leaderboard guardrail (full); pillar_moderate ~21% leak is the realistic-set proxy;
pillar_free ~1% leak is the honest floor. LLM off (inert on score, deterministic, network-free).

Usage:  python -u scripts/validate_satisfaction.py
"""
from __future__ import annotations

import time

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src.agent import Agent

CATALOG = "data/catalog.jsonl"
SETS = [
    ("public",          "data/public_set.jsonl",      None),
    ("pillar_moderate", "data/pillar_moderate.jsonl", None),
    ("pillar_free",     "data/pillar_free.jsonl",     None),
]
# (label, use_satisfaction, pop_weight)
CONFIGS = [
    ("default (coverage+floor)", False, 0.0),
    ("satisfaction pop=0.0",     True,  0.0),
    ("satisfaction pop=0.15",    True,  0.15),
    ("satisfaction pop=0.3",     True,  0.3),
]


def main() -> None:
    for f in ("USE_LLM_SLOTS", "USE_LLM_INFERENCE", "USE_LLM_RESPONSE", "USE_LLM_RERANK"):
        setattr(Agent, f, False)
    cat_ids, cats, prods = catalog_index(CATALOG)
    loaded = {lbl: load_jsonl(p)[: (n if n else None)] for lbl, p, n in SETS}
    agent = Agent(CATALOG)

    header = f"{'config':>26} | " + " | ".join(f"{lbl:>16}" for lbl, _, _ in SETS)
    print("Fix #1 — satisfaction ranker validation (score / hit / mttc)", flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for label, use_sat, pw in CONFIGS:
        agent.USE_SATISFACTION_RANKER = use_sat
        agent.SATISFACTION_POP_WEIGHT = pw
        agent._satisfaction.pop_weight = pw
        cells = []
        for lbl, _, _ in SETS:
            t0 = time.time()
            r = evaluate(agent, loaded[lbl], cat_ids, cats, prods)
            cells.append(f"{r['recommended_technical_score']:.3f}/{r['hit_rate_at_10']:.2f}/"
                         f"{r['mttc']:.1f}")
        print(f"{label:>26} | " + " | ".join(f"{c:>16}" for c in cells) +
              f"  ({time.time()-t0:.0f}s last)", flush=True)

    print("\nAdopt the satisfaction config that lifts pillar_moderate/free while public stays "
          ">= ~0.902 (within 0.015 of 0.9172).", flush=True)


if __name__ == "__main__":
    main()
