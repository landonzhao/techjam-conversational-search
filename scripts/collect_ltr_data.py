"""Collect leak-balanced LTR training data (docs/ADVANCED_RANKING_PLAN.md, step 2).

Replays public (leaky) + pillar_moderate + pillar_free (honest) through the evaluator loop. At each
session's final turn it captures the ranking signals the agent computed and writes one row per pool
candidate: the RankingFeatures vector + is_target label + leak level. Training on the mix (not public
alone) is what keeps the learned weights from overfitting the verbatim leak.

Output: cache/ltr_data.jsonl

Usage:  python -u scripts/collect_ltr_data.py [--per-set 150]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply, initial_message,
    load_jsonl, materialize_hidden_fields, normalize_recommendations,
)
from src.agent import Agent
from src.ranking_features import FEATURE_NAMES, RankingFeatures

SETS = [
    ("leaky", "data/public_set.jsonl"),
    ("moderate", "data/pillar_moderate.jsonl"),
    ("free", "data/pillar_free.jsonl"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-set", type=int, default=150)
    ap.add_argument("--out", default="cache/ltr_data.jsonl")
    ap.add_argument("--sets", default="leaky,moderate,free",
                    help="comma-separated leak levels to collect")
    args = ap.parse_args()
    wanted = set(args.sets.split(","))
    sets = [(leak, path) for leak, path in SETS if leak in wanted]

    for f in ("USE_LLM_SLOTS", "USE_LLM_INFERENCE", "USE_LLM_RESPONSE", "USE_LLM_RERANK"):
        setattr(Agent, f, False)
    cat_ids, cats, prods = catalog_index("data/catalog.jsonl")
    agent = Agent("data/catalog.jsonl")
    feats = RankingFeatures(agent._catalog.products, agent._coverage)

    # capture the pool + the score dicts the agent computes each turn
    cap: dict = {}
    orig_sat = agent._satisfaction.rank

    def wrap_sat(asins, phrases):
        order, sc = orig_sat(asins, phrases)
        cap["asins"] = list(asins)        # incoming (retrieval-fused) order, pre-satisfaction sort
        cap["sat"] = sc
        return order, sc
    agent._satisfaction.rank = wrap_sat

    ce = agent._cross_encoder
    orig_ce = ce.scores if ce is not None else None
    if ce is not None:
        def wrap_ce(query, asins, depth):
            sc = orig_ce(query, asins, depth)
            cap["ce"] = {a: sc[i] for i, a in enumerate(asins[:len(sc)])} if sc else {}
            return sc
        ce.scores = wrap_ce

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_written = n_sessions = 0
    with out.open("w", encoding="utf-8") as fh:
        for leak, path in sets:
            for s in load_jsonl(path)[: args.per_set]:
                sid = f"ltr_{leak}_{n_sessions}"
                agent.reset(sid, s["user_profile"])
                target = str(s["ground_truth"]["parent_asin"])
                eic, eb = materialize_hidden_fields(s, prods)
                es = {**s, "intent_card": eic, "behavior": eb}
                disclosed: set[str] = set()
                bu = False
                msg = initial_message(es, coarse_category(cats.get(target, [])), disclosed)
                cap.clear()
                budget = None
                category = None
                for turn in range(1, MAX_TURNS + 1):
                    r = agent.respond(sid, msg, turn, TOP_K)
                    st = agent._sessions[sid]
                    category = st.need.category
                    recs = normalize_recommendations(r.get("recommendations"), cat_ids)
                    if target in recs or turn == MAX_TURNS:
                        break
                    msg, bu = customer_reply(es, r.get("ask_attribute"), disclosed, bu)
                n_sessions += 1
                asins = cap.get("asins")
                if not asins or target not in asins:
                    continue    # target not in pool -> no useful ranking label
                st = agent._sessions[sid]
                import re as _re
                for c in reversed(st.need.positives("budget")):
                    nums = _re.findall(r"\d+(?:\.\d+)?", c.value)
                    if nums:
                        budget = (float(nums[0]) + float(nums[-1])) / 2
                        break
                fv = feats.extract(
                    asins, phrases=list(st.constraint_phrases),
                    satisfaction_scores=cap.get("sat"), ce_scores=cap.get("ce"),
                    budget=budget, category=st.need.category)
                for a in asins:
                    fh.write(json.dumps({
                        "f": fv[a], "y": int(a == target), "leak": leak, "sid": sid}) + "\n")
                    rows_written += 1

    print(f"features: {FEATURE_NAMES}")
    print(f"wrote {rows_written} rows from {n_sessions} sessions -> {out}")


if __name__ == "__main__":
    main()
