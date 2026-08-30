"""Oracle pass on the honest (leak-free) set — split failures into RETRIEVAL vs RANKING.

For every honest-test session we replay the evaluator's exact turn loop, but each turn we also peek
at the bot's SEARCH POOL (the ~200 candidates retrieval pulls, before ranking). Then:

  retrieval recall  = fraction of sessions where the target ever entered the pool. This is
                      config-independent (the pool is built before ranking) and answers the core
                      question: does our understanding/search find the target on REWORDED language?

  among the MISSES (target not in top-10 with our BEST ranking config):
    - target WAS in the pool  -> RANKING's fault  (found it, ranked it away — recoverable headroom)
    - target NOT in the pool  -> RETRIEVAL/understanding's fault (never found — a different fix)

Ranking config = the current default (P2 dual-track), so "misses" are the residual after the active
submission ranking path. Runtime exceptions are reported separately from retrieval/ranking faults.

Usage:  python -u scripts/oracle_leakfree.py
"""
from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply, initial_message,
    load_jsonl, materialize_hidden_fields, normalize_recommendations,
)
from scripts.eval_support import new_isolated_agent
from src.agent import Agent

CATALOG = "data/catalog.jsonl"
LEAKFREE = "data/language_stress_set.jsonl"


def main() -> None:
    cat_ids, cats, prods = catalog_index(CATALOG)
    rows = load_jsonl(LEAKFREE)

    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    agent = new_isolated_agent(CATALOG)
    # Legacy coverage knobs are retained for comparison; P2 is the current default path.
    agent.COVERAGE_RETRIEVAL_WEIGHT = 2.0
    agent.COVERAGE_INFORMATIVE_MIN = 0.5
    agent.COVERAGE_DISCRIMINATION_PCTL = 0.9
    agent.SUPPRESS_POP_ON_PARAPHRASE = True

    # instrument retrieval to capture the pool each turn (wrap the bound method with a plain fn)
    orig_retrieve = agent._retrieve
    captured: dict[str, list[str]] = {"pool": []}

    def wrapped(state, pool):
        res = orig_retrieve(state, pool)
        captured["pool"] = list(res)
        return res

    agent._retrieve = wrapped

    n = hits = in_pool_ever = miss_in_pool = miss_not_in_pool = 0
    runtime_error_sessions = runtime_errors = 0
    error_types: Counter[str] = Counter()
    best_ranks_on_miss: list[int] = []
    pool_sizes: list[int] = []

    for i, sample in enumerate(rows):
        n += 1
        sid = f"oracle_{i}"
        agent.reset(sid, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        eic, eb = materialize_hidden_fields(sample, prods)
        es = {**sample, "intent_card": eic, "behavior": eb}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(es, coarse_category(cats.get(target, [])), disclosed)

        hit = False
        session_errored = False
        target_in_pool = False
        best_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(sid, user_message, turn, TOP_K)
            except Exception as exc:
                session_errored = True
                runtime_errors += 1
                error_types[type(exc).__name__] += 1
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            poolids = captured["pool"]
            pool_sizes.append(len(poolids))
            if target in poolids:
                target_in_pool = True
                r = poolids.index(target) + 1
                best_rank = r if best_rank is None else min(best_rank, r)
            ranked = normalize_recommendations(response.get("recommendations"), cat_ids)
            if override_applied and target in ranked:
                hit = True
                break
            if turn == MAX_TURNS:
                break
            override = es.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                nv = str(override.get("new_value", ""))
                if nv:
                    disclosed.add(nv)
                user_message = str(override.get("message", "Actually, ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    es, response.get("ask_attribute"), disclosed, boundary_used)

        if target_in_pool:
            in_pool_ever += 1
        if hit:
            hits += 1
        elif session_errored:
            # Do not call a swallowed agent exception a ranking fault merely because retrieval had
            # already found the target. The official evaluator still scores it as a miss, but this
            # diagnostic must expose the actual subsystem responsible.
            runtime_error_sessions += 1
        elif target_in_pool:
            miss_in_pool += 1
            if best_rank is not None:
                best_ranks_on_miss.append(best_rank)
        else:
            miss_not_in_pool += 1

    misses = n - hits
    pct = lambda x, d: (100.0 * x / d) if d else 0.0
    print(f"ORACLE — honest (leak-free) set, {n} sessions, best ranking config, LLM off", flush=True)
    print(f"avg pool size ~{statistics.mean(pool_sizes):.0f}", flush=True)
    print("-" * 68, flush=True)
    print(f"RETRIEVAL RECALL (target ever in pool):  {in_pool_ever}/{n}  = {pct(in_pool_ever,n):.1f}%",
          flush=True)
    print(f"end-to-end HIT@10 (best ranking):        {hits}/{n}  = {pct(hits,n):.1f}%", flush=True)
    print("-" * 68, flush=True)
    print(f"MISSES: {misses}", flush=True)
    print(f"  AGENT runtime errors:                  {runtime_error_sessions}/{misses} = "
          f"{pct(runtime_error_sessions,misses):.1f}%  of misses "
          f"({runtime_errors} failed turns; {dict(error_types)})", flush=True)
    print(f"  RANKING's fault  (in pool, ranked away): {miss_in_pool}/{misses} = "
          f"{pct(miss_in_pool,misses):.1f}%  of misses", flush=True)
    print(f"  RETRIEVAL's fault (never in pool):       {miss_not_in_pool}/{misses} = "
          f"{pct(miss_not_in_pool,misses):.1f}%  of misses", flush=True)
    if best_ranks_on_miss:
        med = statistics.median(best_ranks_on_miss)
        print(f"  on ranking-fault misses, target's best pool rank: median {med:.0f} "
              f"(min {min(best_ranks_on_miss)}, max {max(best_ranks_on_miss)})", flush=True)
    print("-" * 68, flush=True)
    print("Read: retrieval recall = how well understanding/search handles reworded language.", flush=True)
    print("      ranking-fault share of misses = headroom a better ranker can still recover.", flush=True)


if __name__ == "__main__":
    main()
