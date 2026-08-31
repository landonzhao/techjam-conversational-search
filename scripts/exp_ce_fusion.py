"""Ranking-isolation harness for the cross-encoder fusion experiment (roadmap component I).

WHAT THIS ISOLATES
------------------
Retrieval is held fixed. For every honest-set session we replay the official disclosure loop and, at
each turn, capture the EXACT pre-cross-encoder candidate pool (the satisfaction-ranked order the CE
scores), the per-candidate satisfaction score, and the cross-encoder score. We then re-rank that SAME
pool under several fusion strategies and measure the ground-truth target's rank under each. Because
every strategy sees the identical captured pool, any metric delta is attributable to fusion alone —
retrieval can neither be credited nor blamed.

Capture point: we wrap `Agent._cross_encoder.scores`, which the agent calls with `base = list(
candidates)` (the satisfaction-ordered pool) and `state.query_text()` — exactly the fusion input.
Satisfaction scores are stashed from `Agent._satisfaction.rank` in the same turn.

STRATEGIES
----------
* rrf        — EXACT replica of the current production fusion (rank-only RRF of the CE order,
               CE_WEIGHT, RRF_K). This is the baseline.
* beta=b     — calibrated convex combination: FinalScore(c) = (1-b)*SatNorm(c) + b*CENorm(c) over the
               CE head (min-max normalized), tail kept in satisfaction order. b=0 ~ satisfaction only,
               b=1 ~ CE only.
* sat        — satisfaction order only (no CE); an additional reference point.

METRICS (per strategy, over in-pool sessions)
---------------------------------------------
First-appearance (evaluator-faithful, full pool, no reveal hold-back):
  MRR, Hit@10, MTTC-proxy (first turn target reaches top-10).
Best-rank distribution (min target rank across turns): Hit@1/@3/@10, median/mean rank.
Head-to-head vs rrf: % improved / unchanged / worsened by best rank.
Latency: CE scoring time and fusion overhead.

Usage:
  python -u scripts/exp_ce_fusion.py --set language_stress_set
  python -u scripts/exp_ce_fusion.py --set pillar_free --limit 120 --betas 0,0.2,0.4,0.5,0.6,0.8,1.0
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply, initial_message,
    load_jsonl, materialize_hidden_fields,
)
from src.agent import Agent
from src.config import RRF_K
from src.retrieval import rrf

CATALOG = "data/catalog.jsonl"


# --------------------------------------------------------------------------- fusion strategies
def _minmax(vals: list[float]) -> list[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1e-12:
        return [0.5] * len(vals)          # degenerate → neutral (preserves incoming tie-break)
    return [(v - lo) / (hi - lo) for v in vals]


def fuse_rrf(pool: list[str], sat: dict, ce: list[float],
             ce_weight: float = 1.0, k: int = RRF_K) -> list[str]:
    """Exact replica of Agent.respond's current CE fusion (rank-only RRF)."""
    if not ce:
        return list(pool)
    head = pool[:len(ce)]
    ce_order_idx = sorted(range(len(head)), key=lambda i: -ce[i])
    secondary = [head[i] for i in ce_order_idx]
    return rrf(pool, secondary, ce_weight, k, top_n=len(pool))


def fuse_convex(pool: list[str], sat: dict, ce: list[float], beta: float) -> list[str]:
    """Calibrated convex combination over the CE head; tail kept in satisfaction order.
    FinalScore(c) = (1-beta)*SatNorm(c) + beta*CENorm(c). Stable tie-break on incoming head index."""
    if not ce:
        return list(pool)
    d = len(ce)
    head = pool[:d]
    sat_n = _minmax([float(sat.get(a, 0.0)) for a in head])
    ce_n = _minmax(list(ce))
    blended = [(1.0 - beta) * sat_n[i] + beta * ce_n[i] for i in range(d)]
    order_idx = sorted(range(d), key=lambda i: (-blended[i], i))   # deterministic
    return [head[i] for i in order_idx] + list(pool[d:])


def fuse_sat(pool: list[str], sat: dict, ce: list[float]) -> list[str]:
    return list(pool)   # incoming pool IS the satisfaction order


# --------------------------------------------------------------------------- capture + replay
def run(dataset: str, limit: int | None, betas: list[float]) -> None:
    cat_ids, cats, prods = catalog_index(CATALOG)
    rows = load_jsonl(dataset)
    if limit:
        rows = rows[:limit]

    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    Agent.USE_CROSS_ENCODER = True        # need the CE model loaded to score
    agent = Agent(CATALOG)
    if agent._cross_encoder is None:
        raise SystemExit("Cross-encoder unavailable — cannot run this experiment.")

    # --- instrument: stash satisfaction scores, capture CE scoring inputs/outputs ---
    turn_cap: dict = {"sat": {}, "records": []}
    orig_rank = agent._satisfaction.rank
    orig_scores = agent._cross_encoder.scores
    ce_time = {"t": 0.0, "pairs": 0}

    def wrapped_rank(asins, phrases):
        order, sat = orig_rank(asins, phrases)
        turn_cap["sat"] = sat
        return order, sat

    def wrapped_scores(query, asins, depth):
        t0 = time.perf_counter()
        ce = orig_scores(query, asins, depth)
        ce_time["t"] += time.perf_counter() - t0
        ce_time["pairs"] += min(len(asins), depth)
        turn_cap["records"].append(
            {"query": query, "pool": list(asins), "ce": list(ce), "sat": dict(turn_cap["sat"])})
        return ce

    agent._satisfaction.rank = wrapped_rank
    agent._cross_encoder.scores = wrapped_scores

    strategies: dict[str, callable] = {"rrf": fuse_rrf, "sat": fuse_sat}
    for b in betas:
        strategies[f"beta={b:g}"] = (lambda pool, sat, ce, _b=b: fuse_convex(pool, sat, ce, _b))

    # per-strategy accumulators
    first_rr = {s: [] for s in strategies}       # reciprocal rank at first top-10 appearance (0 if none)
    first_hit = {s: 0 for s in strategies}
    first_ttc = {s: [] for s in strategies}
    best_rank = {s: [] for s in strategies}      # min target rank across turns (in-pool sessions)
    n = in_pool = 0
    fusion_time = 0.0

    for i, sample in enumerate(rows):
        n += 1
        sid = f"ce_{i}"
        agent.reset(sid, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        eic, eb = materialize_hidden_fields(sample, prods)
        es = {**sample, "intent_card": eic, "behavior": eb}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(es, coarse_category(cats.get(target, [])), disclosed)

        turn_cap["records"] = []
        stall = 0
        for turn in range(1, MAX_TURNS + 1):
            turn_cap["sat"] = {}
            try:
                response = agent.respond(sid, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            # advance disclosure exactly like the evaluator
            if turn == MAX_TURNS:
                break
            override = es.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                nv = str(override.get("new_value", ""))
                if nv:
                    disclosed.add(nv)
                user_message = str(override.get("message", "Actually, ignore my earlier preference."))
                stall = 0
            else:
                new_msg, boundary_used = customer_reply(
                    es, response.get("ask_attribute"), disclosed, boundary_used)
                # stop once disclosure is exhausted (message carries no new constraint)
                if "what matters is" not in new_msg.lower() and "key requirement" not in new_msg.lower():
                    stall += 1
                else:
                    stall = 0
                user_message = new_msg
            if stall >= 1 and override_applied:
                break

        recs = turn_cap["records"]
        # is the target ever in a captured pool?
        session_in_pool = any(target in r["pool"] for r in recs)
        if session_in_pool:
            in_pool += 1

        for s, fn in strategies.items():
            best = None
            first = None
            for t_idx, r in enumerate(recs, start=1):
                if target not in r["pool"]:
                    continue
                t0 = time.perf_counter()
                order = fn(r["pool"], r["sat"], r["ce"])
                fusion_time += time.perf_counter() - t0
                rank = order.index(target) + 1
                best = rank if best is None else min(best, rank)
                if first is None and rank <= TOP_K:
                    first = (t_idx, rank)
            if session_in_pool:
                best_rank[s].append(best if best is not None else 10_000)
            if first is not None:
                first_hit[s] += 1
                first_rr[s].append(1.0 / first[1])
                first_ttc[s].append(first[0])
            else:
                first_rr[s].append(0.0)

    # ----------------------------------------------------------------- report
    print(f"\n=== CE FUSION ISOLATION — {Path(dataset).stem}  ({n} sessions, {in_pool} target-in-pool) ===")
    print(f"CE: {ce_time['pairs']} pairs scored in {ce_time['t']:.1f}s "
          f"({ce_time['pairs']/max(ce_time['t'],1e-9):.0f} pairs/s); "
          f"fusion overhead total {fusion_time*1000:.0f}ms")
    print(f"{'strategy':<12} {'MRR':>7} {'Hit@10':>7} {'Hit@3':>7} {'Hit@1':>7} "
          f"{'medRank':>8} {'meanRank':>9} {'MTTC*':>7}  vs-rrf(+/=/-)")
    rrf_best = best_rank["rrf"]
    def hit_at(ranks, k):
        v = [r for r in ranks if r < 10_000]
        return sum(1 for r in v if r <= k) / len(v) if v else 0.0
    for s in strategies:
        rr = first_rr[s]
        mrr = statistics.fmean(rr) if rr else 0.0
        valid = [r for r in best_rank[s] if r < 10_000]
        med = statistics.median(valid) if valid else float("nan")
        mean = statistics.fmean(valid) if valid else float("nan")
        h10 = hit_at(best_rank[s], 10)
        h3 = hit_at(best_rank[s], 3)
        h1 = hit_at(best_rank[s], 1)
        ttc = statistics.fmean(first_ttc[s]) if first_ttc[s] else float("nan")
        imp = eq = wor = 0
        for a, b in zip(best_rank[s], rrf_best):
            if a < b:
                imp += 1
            elif a > b:
                wor += 1
            else:
                eq += 1
        print(f"{s:<12} {mrr:>7.4f} {h10:>7.3f} {h3:>7.3f} {h1:>7.3f} "
              f"{med:>8.1f} {mean:>9.1f} {ttc:>7.2f}  {imp:>3}/{eq:>3}/{wor:>3}")
    print("MRR/Hit@10/MTTC* = first top-10 appearance across turns (full pool, no reveal hold-back).")
    print("medRank/meanRank/Hit@k = best (min) target rank across turns, in-pool sessions only.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="language_stress_set")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--betas", default="0,0.2,0.4,0.5,0.6,0.8,1.0")
    args = ap.parse_args()
    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    run(f"data/{args.set}.jsonl", args.limit, betas)


if __name__ == "__main__":
    main()
