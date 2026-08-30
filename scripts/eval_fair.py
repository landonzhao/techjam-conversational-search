"""Fair head-to-head evaluator — frozen dataset ruler.

Uses the exact same two datasets used to establish the integration baseline on 2026-08-31:

    public_set.jsonl          md5 0801ae47d6efefd557a7ea4a598c9da3
    language_stress_set.jsonl md5 263b2fd7777ccc0d022875acd980e3db  (main-branch version)

Do NOT regenerate these files when comparing branches — use this script so every delta
is measured against the same frozen ruler.

Integration baseline (consolidated-ranking, clean cache, LLM off):
    PUBLIC  Tech 0.8842  Hit@10 0.955  MRR 0.8333  MTTC 3.17
    HONEST  Tech 0.7350  Hit@10 0.784  MRR 0.6800  MTTC 4.05

Floor rule: public TechnicalScore must stay >= 0.88 for any shipping candidate.

Usage:
    python -u scripts/eval_fair.py
    python -u scripts/eval_fair.py --skip-honest     # public only, faster CI check
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src.agent import Agent

CATALOG = "data/catalog.jsonl"
PUBLIC_SET = "data/public_set.jsonl"
HONEST_SET = "data/language_stress_set.jsonl"

# Checksums for the frozen ruler datasets.
EXPECTED_MD5 = {
    PUBLIC_SET: "0801ae47d6efefd557a7ea4a598c9da3",
    HONEST_SET: "263b2fd7777ccc0d022875acd980e3db",
}

PUBLIC_FLOOR = 0.88


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_datasets() -> None:
    """Warn loudly if the eval datasets have drifted from the frozen ruler."""
    ok = True
    for path, expected in EXPECTED_MD5.items():
        actual = _md5(path)
        if actual != expected:
            print(f"WARNING: {path} checksum mismatch — results not comparable to baseline!\n"
                  f"  expected {expected}\n  actual   {actual}", flush=True)
            ok = False
    if not ok:
        print("  Regenerated datasets break cross-branch comparability. "
              "Use the frozen files from main.", flush=True)


def show(name: str, r: dict) -> None:
    print(f"\n{name}: TechnicalScore {r['recommended_technical_score']:.4f}  "
          f"(hit@10 {r['hit_rate_at_10']:.3f}  mrr {r['mrr']:.4f}  mttc {r['mttc']:.2f})",
          flush=True)
    for scen, m in sorted(r.get("scenario_metrics", {}).items()):
        print(f"    {scen:>16}: hit {m['hit_rate_at_10']:.3f}  mrr {m['mrr']:.4f}  mttc {m['mttc']:.2f}",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-honest", action="store_true",
                        help="Skip the honest set (faster public-only guardrail check)")
    args = parser.parse_args()

    _check_datasets()

    cat_ids, cats, prods = catalog_index(CATALOG)
    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    agent = Agent(CATALOG)
    print("FAIR EVAL — frozen ruler (integration-fusion branch)", flush=True)
    print(f"  satisfaction={agent.USE_SATISFACTION_RANKER}  "
          f"guard={agent.USE_RETRIEVAL_GUARD}(k={agent.RETRIEVAL_GUARD_K})  "
          f"regime={getattr(agent, 'USE_REGIME_ROUTING', False)}  "
          f"ce_convex={agent.USE_CE_CONVEX}", flush=True)

    t0 = time.time()
    pub = evaluate(agent, load_jsonl(PUBLIC_SET), cat_ids, cats, prods)
    show("PUBLIC (leaderboard)", pub)

    pub_tech = pub["recommended_technical_score"]
    if pub_tech < PUBLIC_FLOOR:
        print(f"\n  *** FLOOR VIOLATION: {pub_tech:.4f} < {PUBLIC_FLOOR} — do not ship ***",
              flush=True)
    else:
        print(f"\n  floor OK ({pub_tech:.4f} >= {PUBLIC_FLOOR})", flush=True)

    if not args.skip_honest:
        show("HONEST (leak-free)", evaluate(agent, load_jsonl(HONEST_SET), cat_ids, cats, prods))

    print(f"\n({time.time()-t0:.0f}s)", flush=True)
    if pub_tech < PUBLIC_FLOOR:
        sys.exit(1)


if __name__ == "__main__":
    main()
