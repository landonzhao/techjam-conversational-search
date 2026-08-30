"""Replay harness for the shadow private-evaluation suite (docs/SHADOW_EVAL_DESIGN.md).

NON-PRODUCTION. Plays each session's *scripted* user turns through the Agent (unlike the official
evaluator, which synthesizes turns from an intent_card and so cannot express revision / intent
evolution / distractor sentences / paired clarify cases). Only `turns[].user` is fed to the agent;
all other fields are evaluation metadata.

Metrics mirror the official evaluator: freeze at the FIRST turn the target enters top-10.
  MRR = mean(1/first_rank)     Hit@k from that first rank     MTTC = mean(first_hit_turn else 11)
Also attributes each miss to RETRIEVAL (target never in pool) vs RANKING (in pool, ranked away), by
capturing the candidate pool per turn.

Usage:
  python -u scripts/replay_shadow.py --file data/shadow/teaser.jsonl
  python -u scripts/replay_shadow.py --file data/shadow/teaser.jsonl --tag semantic
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, normalize_recommendations
from src.agent import Agent

CATALOG = "data/catalog.jsonl"
TOP_K = 10


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def run(path: str, tag: str | None, flags: dict | None = None) -> None:
    cat_ids, _cats, _prods = catalog_index(CATALOG)
    sessions = load(path)
    if tag:
        sessions = [s for s in sessions
                    if tag in (s.get("tags") or []) or s.get("primary_capability", "").startswith(tag)]

    Agent.USE_LLM_SLOTS = Agent.USE_LLM_INFERENCE = Agent.USE_LLM_RESPONSE = Agent.USE_LLM_RERANK = False
    agent = Agent(CATALOG)
    for k, v in (flags or {}).items():
        setattr(agent, k, v)

    # capture the retrieval pool each turn to split retrieval vs ranking failures
    orig_retrieve = agent._retrieve
    cap = {"pool": []}

    def wrapped(state, pool):
        res = orig_retrieve(state, pool)
        cap["pool"] = list(res)
        return res
    agent._retrieve = wrapped

    rows = []
    for s in sessions:
        sid = s["session_id"]
        target = str(s["target_asin"])
        agent.reset(sid, s.get("user_profile") or {})
        first_rank = first_turn = None
        ever_in_pool = False
        best_pool_rank = None
        for t in s["turns"]:
            resp = agent.respond(sid, t["user"], t["turn"], TOP_K)
            if target in cap["pool"]:
                ever_in_pool = True
                pr = cap["pool"].index(target) + 1
                best_pool_rank = pr if best_pool_rank is None else min(best_pool_rank, pr)
            ranked = normalize_recommendations(resp.get("recommendations"), cat_ids)
            if first_rank is None and target in ranked:
                first_rank = ranked.index(target) + 1
                first_turn = t["turn"]
                break
        fault = ("" if first_rank is not None
                 else ("RANKING" if ever_in_pool else "RETRIEVAL"))
        rows.append({
            "id": sid, "diff": s.get("difficulty", "?"), "cap": s.get("primary_capability", "?"),
            "turns": len(s["turns"]), "hit": first_rank is not None, "rank": first_rank,
            "ttc": first_turn, "rr": (1.0 / first_rank) if first_rank else 0.0,
            "pool_rank": best_pool_rank, "fault": fault,
        })

    # ---- report ----
    n = len(rows)
    mrr = statistics.fmean(r["rr"] for r in rows) if rows else 0.0
    hits = [r for r in rows if r["hit"]]
    hit10 = len(hits) / n if n else 0.0
    hit3 = sum(1 for r in hits if r["rank"] <= 3) / n if n else 0.0
    hit1 = sum(1 for r in hits if r["rank"] == 1) / n if n else 0.0
    mttc = statistics.fmean(r["ttc"] if r["ttc"] else 11 for r in rows) if rows else 0.0
    print(f"\n=== SHADOW REPLAY — {Path(path).stem}  ({n} sessions{', tag='+tag if tag else ''}) ===")
    print(f"{'id':<12}{'diff':<12}{'cap':<18}{'T':>2} {'hit':>4} {'rank':>5} {'pool':>5} {'fault':>9}")
    for r in rows:
        print(f"{r['id']:<12}{r['diff']:<12}{r['cap']:<18}{r['turns']:>2} "
              f"{'Y' if r['hit'] else 'n':>4} {str(r['rank'] or '-'):>5} "
              f"{str(r['pool_rank'] or '-'):>5} {r['fault']:>9}")
    print("-" * 66)
    print(f"MRR {mrr:.4f} | Hit@1 {hit1:.3f} Hit@3 {hit3:.3f} Hit@10 {hit10:.3f} | MTTC {mttc:.2f}")
    # breakdowns
    by = defaultdict(list)
    for r in rows:
        by[("diff", r["diff"])].append(r); by[("cap", r["cap"])].append(r)
    print("by difficulty:", {k[1]: round(statistics.fmean(x["rr"] for x in v), 3)
                             for k, v in sorted(by.items()) if k[0] == "diff"})
    miss = [r for r in rows if not r["hit"]]
    if miss:
        print(f"misses: {len(miss)}  (RANKING {sum(1 for r in miss if r['fault']=='RANKING')}, "
              f"RETRIEVAL {sum(1 for r in miss if r['fault']=='RETRIEVAL')})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/shadow/teaser.jsonl")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--flags", default=None,
                    help="comma list like USE_CATEGORY_GATE=1,USE_CE_CONVEX=1,CE_CONVEX_GATE_MARGIN=0.5")
    args = ap.parse_args()
    flags = {}
    for kv in (args.flags or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                v = float(v) if ("." in v) else int(v)
            except ValueError:
                pass
            flags[k.strip()] = bool(v) if k.strip().startswith("USE_") else v
    run(args.file, args.tag, flags)


if __name__ == "__main__":
    main()
