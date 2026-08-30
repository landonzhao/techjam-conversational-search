"""Stack fix #2 (paraphrase pop-suppression) on top of the conditional floor — P0 follow-up.

The floor (exp_conditional_floor.py) rescues paraphrased turns by fusing the retrieval order back in.
But on those turns coverage_order still collapses toward popularity before the fusion. Suppressing
popularity on exactly the uninformative turns makes coverage_order fall back to the retrieval order,
which the floor then reinforces — the two should compound on leak-free without touching public.

Run AFTER exp_conditional_floor.py frees the CPU (per-turn query embedding is the bottleneck; two
eval jobs thrash). Leak-free in full (the failing set); public sampled as a cheap guardrail.

Usage:  python -u scripts/exp_paraphrase_stack.py
"""
from __future__ import annotations

import time

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src.agent import Agent

CATALOG = "data/catalog.jsonl"
PUBLIC_GUARD_N = 100
# (label, informative_min, retrieval_weight, suppress_pop)
CONFIGS = [
    ("gated w=2.0  suppress OFF", 1.0, 2.0, False),   # floor only (fix #1)
    ("gated w=2.0  suppress ON",  1.0, 2.0, True),    # floor + pop-suppression (fix #1 + #2)
    ("gated w=3.0  suppress ON",  1.0, 3.0, True),
]


def main() -> None:
    pub = load_jsonl("data/public_set.jsonl")[:PUBLIC_GUARD_N]
    leak = load_jsonl("data/language_stress_set.jsonl")
    cat_ids, cats, prods = catalog_index(CATALOG)

    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    agent = Agent(CATALOG)

    header = (f"{'config':>26} | {'PUB-'+str(PUBLIC_GUARD_N):>7} {'hit':>6} {'mrr':>7} | "
              f"{'LEAK-FREE':>9} {'hit':>6} {'mrr':>7}")
    print("floor + paraphrase pop-suppression — leak-free=250 (failing set)", flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for label, imin, w, sup in CONFIGS:
        agent.COVERAGE_INFORMATIVE_MIN = imin
        agent.COVERAGE_RETRIEVAL_WEIGHT = w
        agent.SUPPRESS_POP_ON_PARAPHRASE = sup
        t0 = time.time()
        p = evaluate(agent, pub, cat_ids, cats, prods)
        lk = evaluate(agent, leak, cat_ids, cats, prods)
        print(f"{label:>26} | "
              f"{p['recommended_technical_score']:>7.4f} {p['hit_rate_at_10']:>6.3f} {p['mrr']:>7.4f} | "
              f"{lk['recommended_technical_score']:>9.4f} {lk['hit_rate_at_10']:>6.3f} {lk['mrr']:>7.4f} "
              f"  ({time.time()-t0:.0f}s)", flush=True)

    print("\nKeep pop-suppression only if it raises leak-free without dropping the public guardrail.",
          flush=True)


if __name__ == "__main__":
    main()
