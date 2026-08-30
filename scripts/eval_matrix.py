"""Run the current ranker with a relevance-preserving popularity ablation.

The matrix covers leak-free, public, and optional synthetic data in normal and pop-ablated modes.
The ablation zeros every popularity signal without changing candidate-pool size. Use
``scripts/oracle_leakfree.py`` separately for retrieval-versus-ranking headroom analysis.

Usage:
  python -u scripts/eval_matrix.py                 # leak-free + 100 public sessions
  python -u scripts/eval_matrix.py --public-n 200  # full public guardrail
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ranking as ranking_mod
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.eval_support import new_isolated_agent
from src.agent import Agent

CATALOG = "data/catalog.jsonl"

# Historical rankers remain in git history and dedicated experiment scripts. The matrix measures
# only the active configuration so its headline is not confused by dead comparison rows.
CONFIGS = {"dual-track (current)": {"USE_DUAL_TRACK_RANKER": True}}

# (label, path, default sample size or None for full)
SETS = [
    ("leak-free", "data/language_stress_set.jsonl", None),   # primary — always full
    ("public",    "data/public_set.jsonl",          100),    # guardrail — sampled unless --public-n
]
if Path("data/synthetic_set.jsonl").exists():
    SETS.append(("synthetic", "data/synthetic_set.jsonl", 1000))


def apply_config(agent: Agent, cfg: dict) -> None:
    """Apply one matrix arm without carrying its feature flags into the next arm."""
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
            "dual_leaky_popularity_weight": agent.DUAL_LEAKY_POPULARITY_WEIGHT,
        }
        agent.COVERAGE_POP_BLEND = 0.0
        ranking_mod.POP_WEIGHT = 0.0
        ranking_mod.COVERAGE_TIE_BREAK = "base"   # tie-break -> incoming retrieval order
        agent._satisfaction.pop_weight = 0.0
        agent.DUAL_POPULARITY_WEIGHT = 0.0
        agent.DUAL_LEAKY_POPULARITY_WEIGHT = 0.0
    elif hasattr(agent, "_pop_saved"):
        saved = agent._pop_saved
        agent.COVERAGE_POP_BLEND = saved["coverage_blend"]
        ranking_mod.POP_WEIGHT = saved["ranking_pop_weight"]
        ranking_mod.COVERAGE_TIE_BREAK = saved["tie_break"]
        agent._satisfaction.pop_weight = saved["satisfaction_pop_weight"]
        agent.DUAL_POPULARITY_WEIGHT = saved["dual_popularity_weight"]
        agent.DUAL_LEAKY_POPULARITY_WEIGHT = saved["dual_leaky_popularity_weight"]


def score_row(agent: Agent, rows: list, cat_ids, cats, prods) -> tuple:
    r = evaluate(agent, rows, cat_ids, cats, prods)
    return (r["recommended_technical_score"], r["hit_rate_at_10"], r["mrr"], r["mttc"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-n", type=int, default=None, help="public guardrail sample size")
    ap.add_argument("--synth-n", type=int, default=None, help="synthetic diagnostic sample size")
    ap.add_argument("--pool-size", type=int, default=200,
                    help="candidate pool size for every mode/config (default: 200)")
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

    print(f"EVAL MATRIX — score / hit@10 / mrr / mttc  "
          f"(leak-free=PRIMARY, public=GUARDRAIL; pool={args.pool_size})",
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
                s, h, m, t = score_row(agent, loaded[lbl], cat_ids, cats, prods)
                cells.append(f"{s:.4f}/{h:.2f}/{m:.3f}/{t:.2f}")
            print(f"{name:>26} | " + " | ".join(f"{c:>22}" for c in cells), flush=True)
        set_pop_ablation(agent, False)

    print("\nPop-ablated mode zeros popularity while preserving candidate-pool size.", flush=True)


if __name__ == "__main__":
    main()
