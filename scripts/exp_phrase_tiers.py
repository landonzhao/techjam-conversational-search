"""Ablation for the graduated phrase-bonus middle tier (ByteMe-style).

The CoverageReranker scores two phrase tiers: coarse token overlap (floor) and an exact-substring
bonus (COVERAGE_FULL_PHRASE_BONUS). This adds a *middle* tier: a partial bonus when a long
constraint phrase's contiguous leading prefix matches even though the exact phrase does not. A
single altered/inserted trailing word destroys the exact-substring match but leaves a long prefix
intact, so the tier degrades more gracefully and sharpens near-miss ranks (MRR).

Measured result (LLM off, dense at shipped default):

    config | PARAPHR    hit     mrr |    HARD    hit     mrr
       off |  0.6909  0.775  0.5965 |  0.6367  0.714  0.5563
    0.5/25 |  0.6914  0.775  0.5984 |  0.6417  0.717  0.5647   <- shipped default
    0.5/15 |  0.6926  0.775  0.6017 |  0.6374  0.714  0.5581
   0.75/20 |  0.6914  0.775  0.5984 |  0.6383  0.714  0.5604

PUBLIC is byte-identical across all settings and so is omitted from the sweep by default: the
simulator leaks exact substrings, the exact-match tier always dominates, and the prefix tier never
differentiates there. The tier only moves paraphrase/hard, where the exact phrase is absent but a
long prefix survives — i.e. it is a private-set generalization signal, not a public-set lever.
`0.5 bonus / 25 chars` is the balanced pick: best hard, small paraphrase gain, zero public change.

Usage:  PYTHONPATH=. python -u scripts/exp_phrase_tiers.py
        PYTHONPATH=. python -u scripts/exp_phrase_tiers.py --public   # also re-confirm public=neutral
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from evaluator.robustness import run as robrun
from scripts.eval_support import new_isolated_agent
from src.agent import Agent

CATALOG = "data/catalog.jsonl"


def main() -> None:
    with_public = "--public" in sys.argv
    pub = load_jsonl("data/public_set.jsonl")
    hard = [r for r in load_jsonl("data/synthetic_set.jsonl")
            if r.get("difficulty_bucket") == "hard"]
    cat_ids, cats, prods = catalog_index(CATALOG)

    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    agent = new_isolated_agent(CATALOG)

    # (label, use_tiers, prefix_bonus, prefix_chars)
    rows = [
        ("off",     False, 0.0,  25),
        ("0.5/25",  True,  0.50, 25),
        ("0.5/15",  True,  0.50, 15),
        ("0.75/20", True,  0.75, 20),
    ]
    pub_col = f" {'PUBLIC':>7} {'mrr':>7}" if with_public else ""
    header = (f"{'config':>9} |{pub_col}{' |' if with_public else ''} {'PARAPHR':>7} {'hit':>6} "
              f"{'mrr':>7} | {'HARD':>7} {'hit':>6} {'mrr':>7}")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for label, use, pb, pc in rows:
        agent.USE_PHRASE_TIERS = use
        agent.COVERAGE_PREFIX_BONUS = pb
        agent.COVERAGE_PREFIX_CHARS = pc
        t0 = time.time()
        pub_cell = ""
        if with_public:
            p = evaluate(agent, pub, cat_ids, cats, prods)
            pub_cell = f" {p['recommended_technical_score']:>7.4f} {p['mrr']:>7.4f} |"
        r = robrun(agent, pub, cat_ids, cats, prods, "paraphrase")
        h = evaluate(agent, hard, cat_ids, cats, prods)
        print(f"{label:>9} |{pub_cell} {r['score']:>7.4f} {r['hit@10']:>6.3f} {r['mrr']:>7.4f} | "
              f"{h['recommended_technical_score']:>7.4f} {h['hit_rate_at_10']:>6.3f} "
              f"{h['mrr']:>7.4f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\nKeep the tier only if it raises/holds robustness MRR without dropping hit@10 and stays "
          "neutral on public. Shipped default: bonus 0.5 / 25 chars (Agent.USE_PHRASE_TIERS).",
          flush=True)


if __name__ == "__main__":
    main()
