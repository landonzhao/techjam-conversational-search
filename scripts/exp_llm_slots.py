"""Three-arm experiment: does LLM slot extraction help, and what does it cost?

Isolates the slot-extraction variable on the synthetic HARD tier (sparse, low-popularity
targets — where regex is weakest and coverage demotion bites hardest). Every other LLM
feature is held off so token cost is attributable purely to slot extraction.

Arms:
  OFF        regex only                          (LLM_SLOT off)            -> 0 tokens
  GATED      LLM when regex found < 2 slots      (LLM_SLOT_MAX_REGEX = 2)  -> low tokens
  ALL-TURNS  LLM on every substantive turn       (LLM_SLOT_MAX_REGEX = 99) -> high tokens

Reports scored metrics + reported token usage per arm. Live Gemini calls are cached to
cache/llm_slot_cache.json, so re-runs are fast.

Usage:  python scripts/exp_llm_slots.py [tier]      # tier defaults to "hard"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.eval_support import new_isolated_agent
from src.agent import Agent

TIER = sys.argv[1] if len(sys.argv) > 1 else "hard"
SYNTH = "data/synthetic_set.jsonl"
CATALOG = "data/catalog.jsonl"


def run() -> None:
    rows = [r for r in load_jsonl(SYNTH) if r.get("difficulty_bucket") == TIER]
    print(f"tier={TIER!r}  sessions={len(rows)}\n")
    cat_ids, cats, prods = catalog_index(CATALOG)

    # One agent, reused across arms. Hold every non-slot LLM feature off so the only
    # variable is slot extraction and token cost is attributable purely to it.
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    Agent.USE_CROSS_ENCODER = False
    Agent.USE_LLM_SLOTS = True
    agent = new_isolated_agent(CATALOG)
    extractor = agent._slot_extractor  # keep a handle so the OFF arm can null it

    arms = [
        ("OFF", None, 0),
        ("GATED", extractor, 2),
        ("ALL-TURNS", extractor, 99),
    ]

    header = f"{'arm':<11} {'score':>7} {'hit@10':>7} {'mrr':>7} {'mttc':>6} {'eff':>6} {'tokens':>9}"
    print(header)
    print("-" * len(header))
    for name, ext, gate in arms:
        agent._slot_extractor = ext
        agent.LLM_SLOT_MAX_REGEX = gate
        t0 = time.time()
        r = evaluate(agent, rows, cat_ids, cats, prods)
        dt = time.time() - t0
        tok = r["reported_token_usage"]["total_tokens"]
        print(f"{name:<11} {r['recommended_technical_score']:>7.4f} "
              f"{r['hit_rate_at_10']:>7.3f} {r['mrr']:>7.4f} {r['mttc']:>6.2f} "
              f"{r['efficiency']:>6.3f} {tok:>9,}   ({dt:.0f}s)")

    print("\nInterpretation: GATED/ALL-TURNS must BEAT OFF on score to justify the tokens.")
    print("If ALL-TURNS ~= GATED, the <2 gate keeps the win at a fraction of the cost.")


if __name__ == "__main__":
    run()
