"""P0.2 measurement harness — the blocking scoreboard for all ranking work.

One command prints a matrix of {ranking config} x {normal, popularity-ablated} over the three sets:
  - leak-free  (data/language_stress_set.jsonl)  PRIMARY  — generalization, the set we fail on
  - public     (data/public_set.jsonl)           GUARDRAIL — the leaky leaderboard distribution
  - synthetic  (data/synthetic_set.jsonl)         DIAGNOSTIC — harder leaky cases

Two things the plan (docs/STRENGTHENING_PLAN.md P0.2) needs and ad-hoc exp_* scripts don't give:
  1. POP-ABLATED mode (popularity terms zeroed, profile handling and pool unchanged): if a config's
     score survives with popularity removed, it ranks on RELEVANCE, not fame. This is the P1 baseline.
  2. RANKING ORACLE (--oracle): best achievable score if the target, whenever it is anywhere in the
     retrieved pool, were ranked #1. Separates "retrieval never found it" (unfixable by ranking) from
     "found it, ranked it away" (the headroom ranking can recover).

This run also closes P0.1: compare `flat floor` vs `disc gate` on leak-free (pop-ablated) + public.

Usage:
  python -u scripts/eval_matrix.py                 # focused default matrix (~20-30 min, LLM off)
  python -u scripts/eval_matrix.py --public-n 250  # full public guardrail
  python -u scripts/eval_matrix.py --oracle        # add the retrieval/ranking headroom row
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ranking as ranking_mod
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.eval_support import new_isolated_agent
from src.agent import Agent

CATALOG = "data/catalog.jsonl"

# Named ranking configs, each a dict of agent attributes to set. Unset floor/gate attrs default off.
CONFIGS = {
    "no-floor (pure coverage)": dict(COVERAGE_INFORMATIVE_MIN=0.0, COVERAGE_RETRIEVAL_WEIGHT=0.0,
                                     SUPPRESS_POP_ON_PARAPHRASE=False, USE_SATISFACTION_RANKER=False,
                                     USE_DUAL_TRACK_RANKER=False),
    "disc gate + suppress":     dict(COVERAGE_INFORMATIVE_MIN=0.5, COVERAGE_RETRIEVAL_WEIGHT=2.0,
                                     COVERAGE_DISCRIMINATION_PCTL=0.9, SUPPRESS_POP_ON_PARAPHRASE=True,
                                     USE_SATISFACTION_RANKER=False, USE_DUAL_TRACK_RANKER=False),
    "satisfaction (Phase 1)":   dict(USE_SATISFACTION_RANKER=True, USE_DUAL_TRACK_RANKER=False),
    "dual-track (P2)":          dict(USE_DUAL_TRACK_RANKER=True),
}

# (label, path, default sample size or None for full)
SETS = [
    ("leak-free", "data/language_stress_set.jsonl", None),   # primary — always full
    ("public",    "data/public_set.jsonl",          100),    # guardrail — sampled unless --public-n
]
if Path("data/synthetic_set.jsonl").exists():
    SETS.append(("synthetic", "data/synthetic_set.jsonl", 1000))


def apply_config(agent: Agent, cfg: dict) -> None:
    # reset the floor/gate/satisfaction knobs to a known baseline, then apply the named overrides
    agent.COVERAGE_INFORMATIVE_MIN = 0.0
    agent.COVERAGE_RETRIEVAL_WEIGHT = 0.0
    agent.COVERAGE_DISCRIMINATION_PCTL = 0.9
    agent.SUPPRESS_POP_ON_PARAPHRASE = False
    agent.USE_SATISFACTION_RANKER = False
    agent.USE_DUAL_TRACK_RANKER = False
    for k, v in cfg.items():
        setattr(agent, k, v)


def set_pop_ablation(agent: Agent, ablate: bool) -> None:
    """Remove popularity while preserving profile handling and candidate-pool size."""
    if ablate:
        agent._pop_saved = {
            "coverage_blend": agent.COVERAGE_POP_BLEND,
            "ranking_pop_weight": ranking_mod.POP_WEIGHT,
            "tie_break": ranking_mod.COVERAGE_TIE_BREAK,
            "satisfaction_pop_weight": agent._satisfaction.pop_weight,
            "dual_popularity_weight": agent.DUAL_POPULARITY_WEIGHT,
        }
        agent.COVERAGE_POP_BLEND = 0.0
        ranking_mod.POP_WEIGHT = 0.0
        ranking_mod.COVERAGE_TIE_BREAK = "base"   # tie-break -> incoming retrieval order
        agent._satisfaction.pop_weight = 0.0
        agent.DUAL_POPULARITY_WEIGHT = 0.0
    elif hasattr(agent, "_pop_saved"):
        saved = agent._pop_saved
        agent.COVERAGE_POP_BLEND = saved["coverage_blend"]
        ranking_mod.POP_WEIGHT = saved["ranking_pop_weight"]
        ranking_mod.COVERAGE_TIE_BREAK = saved["tie_break"]
        agent._satisfaction.pop_weight = saved["satisfaction_pop_weight"]
        agent.DUAL_POPULARITY_WEIGHT = saved["dual_popularity_weight"]


def score_row(agent: Agent, rows: list, cat_ids, cats, prods) -> tuple:
    r = evaluate(agent, rows, cat_ids, cats, prods)
    return (r["recommended_technical_score"], r["hit_rate_at_10"], r["mrr"],
            r.get("mean_turns_to_conversion", r.get("mttc", 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-n", type=int, default=None, help="public guardrail sample size")
    ap.add_argument("--synth-n", type=int, default=None, help="synthetic diagnostic sample size")
    ap.add_argument("--pool-size", type=int, default=200,
                    help="candidate pool size for every mode/config (default: 200)")
    ap.add_argument("--oracle", action="store_true", help="add retrieval/ranking headroom row")
    args = ap.parse_args()

    cat_ids, cats, prods = catalog_index(CATALOG)
    loaded = {}
    for label, path, default_n in SETS:
        rows = load_jsonl(path)
        n = args.public_n if label == "public" else args.synth_n if label == "synthetic" else None
        n = n if n is not None else default_n
        loaded[label] = rows if (n is None or label == "leak-free") else rows[:n]

    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    agent = new_isolated_agent(CATALOG, pool_size=args.pool_size)

    print(f"EVAL MATRIX — score / hit@10 / mrr  (leak-free=PRIMARY, public=GUARDRAIL; pool={args.pool_size})",
          flush=True)
    for mode in ("normal", "pop-ablated"):
        set_pop_ablation(agent, mode == "pop-ablated")
        print(f"\n### {mode.upper()}", flush=True)
        head = f"{'config':>26} | " + " | ".join(f"{lbl:>22}" for lbl, _, _ in SETS)
        print(head, flush=True)
        print("-" * len(head), flush=True)
        for name, cfg in CONFIGS.items():
            apply_config(agent, cfg)
            cells = []
            for lbl, _, _ in SETS:
                t0 = time.time()
                s, h, m, _ = score_row(agent, loaded[lbl], cat_ids, cats, prods)
                cells.append(f"{s:.4f}/{h:.2f}/{m:.3f}")
            print(f"{name:>26} | " + " | ".join(f"{c:>22}" for c in cells), flush=True)
        set_pop_ablation(agent, False)

    print("\nP0.1 verdict: does 'disc gate + suppress' clear public>=0.928 AND leak-free>=0.50?",
          flush=True)
    print("P1 baseline: the pop-ablated leak-free column is the number NeedSatisfactionScorer beats.",
          flush=True)


if __name__ == "__main__":
    main()
