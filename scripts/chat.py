"""Interactive REPL to talk to the agent by hand — end-to-end manual test.

Type shopper messages; see the agent's clarifying question, ask_attribute,
internal intent/phase/convergence state, and the ranked recommendations (with titles).

Usage:
    python scripts/chat.py
    python scripts/chat.py --top-k 5
    python scripts/chat.py --tags casual,cotton --summary "budget-conscious shopper"

Commands inside the REPL:
    :state     show full internal state (need, belief, DCP context/plan/profile)
    :reset     start a fresh session
    :quit      exit
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tags", default="", help="comma-separated preference tags")
    parser.add_argument("--summary", default="", help="profile summary text")
    args = parser.parse_args()

    print(f"Loading agent from {args.catalog} (index build ~15s) ...")
    agent = Agent(args.catalog)
    profile = {
        "preference_tags": [t.strip() for t in args.tags.split(",") if t.strip()],
        "summary": args.summary,
    }

    session = "chat"
    agent.reset(session, profile)
    turn = 0
    print("\nAgent ready. Type a shopper message (or :state / :reset / :quit).\n")

    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg in (":quit", ":q", ":exit"):
            break
        if msg == ":reset":
            agent.reset(session, profile)
            turn = 0
            print("[session reset]\n")
            continue
        if msg == ":state":
            st = agent._sessions[session]
            print(f"  intent={st.intent}  buying_score={st.buying_score:.2f}  phase={st.phase}")
            print(f"  need model: {st.need.describe()}")
            print(f"  belief: {st.belief.describe()}")
            print(f"  convergence: {st.conv_state}   missing required: {list(st.belief.attr_uncertainty)}")
            # DCP state
            if st.ctx is not None:
                print(f"  DCP context: turn={st.ctx.turn} volatility={st.ctx.volatility:.2f} "
                      f"intent_trace={[round(x, 2) for x in st.ctx.intent_trace[-5:]]}")
            if st.plan is not None:
                print(f"  DCP plan   : dense_w={st.plan.route_weights.get('dense', '?'):.2f} "
                      f"pool={st.plan.pool_size} rerank={st.plan.rerank_stack} "
                      f"action={st.plan.dialogue_action} ask={st.plan.ask_slot}")
            if st.profile is not None:
                print(f"  DCP profile: {len(st.profile.prefs)} durable prefs "
                      f"{[(p.slot, p.value) for p in st.profile.prefs][:6]}")
                active_profile = agent._personalization_profile(st)
                print(f"  personalization profile: {active_profile.get('preference_tags', [])[:8]}")
            if agent._guidance.stats:
                print("  DCP guidance: "
                      "{k: round(v, 3) for k, v in agent._guidance.stats.items()}")
            print(f"  disclosed constraints (active): {st.effective_constraint_phrases()}")
            print(f"  active ledger: {[(c.operation, c.slot, c.value, c.active) for c in st.need.ledger][-12:]}")
            print(f"  effective query: {st.query_text()[:200]}\n")
            continue

        turn += 1
        resp = agent.respond(session, msg, turn, args.top_k)
        st = agent._sessions[session]
        print(f"\nagent> {resp['message']}")
        print(f"       [turn {turn} | ask_attribute={resp['ask_attribute']} | "
              f"intent={st.intent} | phase={st.phase}]")
        recs = resp["recommendations"]
        if not recs:
            print("       (no recommendations)\n")
            continue
        print(f"       top {len(recs)} recommendations:")
        for i, rec in enumerate(recs, 1):
            asin = rec["parent_asin"]
            p = agent.catalog.get(asin, {})
            title = str(p.get("title") or "")[:70]
            rn = p.get("rating_number")
            print(f"         {i}. {asin}  ({rn} reviews)  {title}")
        print()


if __name__ == "__main__":
    main()
