"""Trace browsing failures on pillar_free: is it clarification, ranking, or reveal timing?

For each browsing session we replay the evaluator loop and, per turn, capture: what the agent asked,
what the customer revealed, the target's TRUE rank in the full ranked list, how many items the agent
revealed, and whether the target was shown. Then we classify each MISS:
  - never-in-top10  : ranking never got the target into the top 10 (ranking fault)
  - shown-late-rank : target reached top 10 but only at a poor rank when first revealed (reveal/MRR)
  - extracted?      : were the constraints actually disclosed (clarification working)?

Usage:  python -u scripts/debug_browsing.py [--n 20]
"""
from __future__ import annotations

import argparse

from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply, initial_message,
    load_jsonl, materialize_hidden_fields, normalize_recommendations,
)
from src.agent import Agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    for f in ("USE_LLM_SLOTS", "USE_LLM_INFERENCE", "USE_LLM_RESPONSE", "USE_LLM_RERANK"):
        setattr(Agent, f, False)
    cat_ids, cats, prods = catalog_index("data/catalog.jsonl")
    rows = [s for s in load_jsonl("data/pillar_free.jsonl")
            if s["scenario_type"] == "browsing"][: args.n]
    a = Agent("data/catalog.jsonl")

    # capture the full ranked order the satisfaction ranker produces each turn
    orig = a._satisfaction.rank
    cap: dict = {"order": []}

    def wrapped(asins, phrases):
        order, sc = orig(asins, phrases)
        cap["order"] = order
        return order, sc
    a._satisfaction.rank = wrapped

    hits = 0
    miss_never = miss_late = 0
    disclosed_ok = 0
    for i, s in enumerate(rows):
        sid = f"dbg_{i}"
        a.reset(sid, s["user_profile"])
        target = str(s["ground_truth"]["parent_asin"])
        eic, eb = materialize_hidden_fields(s, prods)
        es = {**s, "intent_card": eic, "behavior": eb}
        n_constraints = len(eic["hard_constraints"]) + len(eic["soft_preferences"])
        disclosed: set[str] = set()
        bu = False
        msg = initial_message(es, coarse_category(cats.get(target, [])), disclosed)

        best_full_rank = 10**9
        hit = False
        first_reveal_rank = None
        for turn in range(1, MAX_TURNS + 1):
            r = a.respond(sid, msg, turn, TOP_K)
            order = cap["order"]
            full_rank = (order.index(target) + 1) if target in order else None
            if full_rank:
                best_full_rank = min(best_full_rank, full_rank)
            recs = normalize_recommendations(r.get("recommendations"), cat_ids)
            if target in recs and first_reveal_rank is None:
                first_reveal_rank = recs.index(target) + 1
            if args.verbose:
                print(f"  t{turn}: ask={r.get('ask_attribute')!r} shown={len(recs)} "
                      f"target_full_rank={full_rank} disclosed={len(disclosed)}/{n_constraints}")
            if target in recs:
                hit = True
                break
            if turn == MAX_TURNS:
                break
            msg, bu = customer_reply(es, r.get("ask_attribute"), disclosed, bu)

        if len(disclosed) >= min(2, n_constraints):
            disclosed_ok += 1
        if hit:
            hits += 1
        elif best_full_rank <= 10:
            miss_late += 1        # ranking DID reach top10 at some turn, but never shown/locked
        else:
            miss_never += 1       # ranking never reached top10 -> ranking fault
        if args.verbose:
            print(f"session {i}: hit={hit} best_full_rank={best_full_rank} "
                  f"disclosed={len(disclosed)}/{n_constraints}\n")

    n = len(rows)
    print(f"\nBROWSING pillar_free, {n} sessions")
    print(f"  hits: {hits}/{n} = {hits/n:.2f}")
    print(f"  constraints disclosed (clarify worked): {disclosed_ok}/{n} = {disclosed_ok/n:.2f}")
    print(f"  MISS - ranking never reached top10:  {miss_never}/{n}")
    print(f"  MISS - reached top10 but not shown/late: {miss_late}/{n}")


if __name__ == "__main__":
    main()
