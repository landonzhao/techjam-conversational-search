"""Demo: Dynamic Context Programming — visible self-evolution across sessions.

Shows three capabilities in one script:

  1. Cross-session guidance learning: the agent learns which clarification questions
     are informative and de-prioritises questions that users keep waving off.

  2. Within-session adaptation: pool size shrinks as belief converges (200→150→100),
     and waved-off slots are skipped immediately without waiting for EMA propagation.

  3. User profile influence: the summary and preference_tags from user_profile bias
     the expansion retrieval track before any constraints are stated.

Usage:
    python scripts/demo_dcp.py
    python scripts/demo_dcp.py --sessions 8   # more sessions for clearer learning curve
    python scripts/demo_dcp.py --quiet         # suppress per-turn output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Agent

CATALOG = "data/catalog.jsonl"

# A "user" who consistently waves off color and style questions but finds use_case informative.
SIMULATED_USER = {
    "preference_tags": ["comfort", "durability"],
    "summary": "Prior purchases emphasize comfort, durability, quality; ratings are usually positive.",
    "average_prior_rating": 4.8,
    "purchase_frequency": "3-4 prior purchases",
    "rating_style": "usually positive",
}

# Scripted turns that simulate a realistic session:
# Turn 1: open browsing query
# Turn 2: discloses a constraint
# Turn 3: waves off color (boundary attr) — the learning signal
SESSION_TURNS = [
    "I'm looking for a comfortable jacket for outdoor activities.",
    "Key requirement is: material should be waterproof.",
    "No preference for color, actually.",
]


def run_session(agent: Agent, session_id: str, quiet: bool = False) -> dict:
    agent.reset(session_id, SIMULATED_USER)
    last_response = {}
    for turn, msg in enumerate(SESSION_TURNS, start=1):
        resp = agent.respond(session_id, msg, turn, top_k=10)
        last_response = resp
        if not quiet:
            print(f"  T{turn}: [{msg[:55]}]")
            print(f"       → ask='{resp['ask_attribute']}'  "
                  f"top={resp['recommendations'][0]['parent_asin'] if resp['recommendations'] else '—'}")
    return last_response


def print_dcp_state(agent: Agent, session_id: str | None = None, label: str = "") -> None:
    state = agent.dcp_state(session_id)
    print(f"\n{'─'*60}")
    print(f"DCP STATE{' — ' + label if label else ''}")
    print(f"{'─'*60}")

    gw = state["guidance_weights"]
    wr = state["waveoff_rates"]
    print("Guidance weights (higher = ask this slot sooner):")
    for slot, w in list(gw.items())[:6]:
        bar = "█" * max(1, int(w * 15))
        waveoff = wr.get(slot, 0.0)
        print(f"  {slot:12s} {bar:<18} {w:.3f}  (waveoff={waveoff:.2f})")

    sess = state.get("session", {})
    if sess:
        print("\nSession state:")
        print(f"  conv_state        = {sess.get('conv_state')}")
        print(f"  belief.confidence = {sess.get('belief_confidence')}")
        print(f"  volatility        = {sess.get('volatility')}")
        print(f"  intent_trace      = {sess.get('intent_trace')}")
        print(f"  session_waveoffs  = {sess.get('session_waveoffs')}")
        expand = sess.get("profile_expansion_terms", [])
        if expand:
            print(f"  profile_expansion = {expand}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=5,
                        help="Number of sessions to simulate (default: 5)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-turn output")
    args = parser.parse_args()

    Agent.USE_LLM_SLOTS = False
    Agent.USE_LLM_INFERENCE = False
    Agent.USE_LLM_RESPONSE = False
    Agent.USE_LLM_RERANK = False
    agent = Agent(CATALOG)

    print("=" * 60)
    print("DCP DEMO — self-evolution across sessions")
    print("=" * 60)
    print(f"\nUser profile: {json.dumps(SIMULATED_USER, indent=None)}")
    print("\nScenario: user consistently waves off COLOR questions.")
    print("Watch the guidance weights shift — color waveoff rate rises,")
    print("color gets deprioritised, agent asks more useful questions.\n")

    print_dcp_state(agent, label="BEFORE (cold — no sessions run yet)")

    for i in range(args.sessions):
        session_id = f"demo_session_{i}"
        if not args.quiet:
            print(f"\n── Session {i + 1} / {args.sessions} ──")
        run_session(agent, session_id, quiet=args.quiet)

    # Run one more session and inspect mid-session state
    final_session = "demo_session_final"
    agent.reset(final_session, SIMULATED_USER)
    agent.respond(final_session, SESSION_TURNS[0], 1, 10)
    agent.respond(final_session, SESSION_TURNS[1], 2, 10)
    agent.respond(final_session, SESSION_TURNS[2], 3, 10)

    print_dcp_state(agent, final_session,
                    label=f"AFTER {args.sessions} sessions (color waved off each time)")

    # Summarize what changed
    state = agent.dcp_state()
    color_waveoff = state["waveoff_rates"].get("color", 0.0)
    color_weight = state["guidance_weights"].get("color", 1.0)
    budget_weight = state["guidance_weights"].get("budget", 1.0)
    usecase_weight = state["guidance_weights"].get("use_case", 1.0)

    print("=" * 60)
    print("WHAT THE AGENT LEARNED:")
    print("=" * 60)
    print(f"  color waveoff rate   : {color_waveoff:.1%}  (was 0%)")
    print(f"  color guidance weight: {color_weight:.3f} (lower = deprioritised)")
    print(f"  budget weight        : {budget_weight:.3f}")
    print(f"  use_case weight      : {usecase_weight:.3f}")
    print()
    if color_waveoff > 0.3:
        print("✓ Agent learned: this user doesn't care about color.")
        print("  Next session: color question will be skipped in favour of")
        print("  more informative slots (use_case, budget, material).")
    else:
        print("  Run with --sessions 10 to see stronger signal accumulation.")
    print()
    print("Pool adaptation (visible in per-turn belief confidence):")
    print("  PROBE  → pool=200 (wide, exploring)")
    print("  CONFIRM→ pool=150 (narrowing)")
    print("  DELIVER→ pool=100 (delivering)")
    print()
    profile_terms = state["session"]["profile_expansion_terms"]
    if profile_terms:
        print(f"Profile expansion active: {profile_terms}")
        print("  These terms from user_profile[summary] bias the")
        print("  initial retrieval query before any constraints are stated.")


if __name__ == "__main__":
    main()
