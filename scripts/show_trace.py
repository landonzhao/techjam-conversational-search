"""Render captured agent traces (trace.jsonl) as a readable narrative.

Use this to debug a specific case: see, turn by turn, what the agent believed,
which retrieval strategy ran, how the candidates were ranked, why it asked what
it asked, whether it held recommendations back, and where the ground-truth
target sat in the ranking.

Usage:
    python scripts/show_trace.py traces/run/trace.jsonl                 # first session
    python scripts/show_trace.py traces/run/trace.jsonl --sample synth_00042
    python scripts/show_trace.py traces/run/trace.jsonl --miss          # only failed cases
    python scripts/show_trace.py traces/run/trace.jsonl --miss --limit 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_sessions(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def session_hit(session: dict) -> bool:
    for turn in session.get("turns", []):
        tr = turn.get("target_rank") or {}
        if tr.get("rank_shown"):
            return True
    return False


def render(session: dict) -> str:
    lines: list[str] = []
    head = (
        f"═══ {session.get('sample_id', session.get('session_id'))}  "
        f"[{session.get('scenario_type', '?')} / {session.get('difficulty_bucket', '?')} / "
        f"{session.get('category_bucket', '?')}]"
    )
    lines.append(head)
    lines.append(f"    target: {session.get('ground_truth')}")
    profile = session.get("user_profile") or {}
    if profile:
        lines.append(
            f"    profile: {profile.get('rating_style', '?')}, "
            f"tags={profile.get('preference_tags')}")
    lines.append(f"    outcome: {'HIT' if session_hit(session) else 'MISS'}")

    for turn in session.get("turns", []):
        stages = turn.get("stages", {})
        lines.append(f"\n  ── turn {turn['turn']} ──")
        lines.append(f"    user: {turn.get('user_message', '')!r}")

        intent = stages.get("intent", {})
        if intent:
            lines.append(
                f"    intent: {intent.get('intent')} "
                f"(buying={intent.get('smoothed_buying_score')})")
        cons = stages.get("constraints", {})
        if cons.get("slots"):
            lines.append(f"    slots: {cons['slots']}")
        if cons.get("constraint_phrases"):
            lines.append(f"    phrases: {cons['constraint_phrases']}")

        retr = stages.get("retrieval", {})
        if retr:
            lines.append(
                f"    retrieval: {retr.get('strategy')} "
                f"(dense_w={retr.get('dense_weight')}, pool={retr.get('pool_size')}, "
                f"got={retr.get('candidates_returned')})")
            if retr.get("expansion_terms"):
                lines.append(f"      expansion: {retr['expansion_terms']}")

        picks = stages.get("ranking", {}).get("top_picks", [])
        if picks:
            lines.append("    top picks:")
            for p in picks:
                lines.append(
                    f"      {p['rank']}. {p['asin']}  cov={p.get('coverage_score')}  "
                    f"{p.get('title', '')}")

        belief = stages.get("belief", {})
        if belief:
            lines.append(
                f"    belief: conf={belief.get('confidence')} "
                f"entropy={belief.get('entropy')} state={belief.get('conv_state')}")

        clar = stages.get("clarification", {})
        resp = stages.get("response", {})
        if resp.get("ask_attribute"):
            lines.append(
                f"    ask: {resp.get('ask_attribute')} "
                f"— {clar.get('info_gain_phrasing') or ''}")

        reveal = stages.get("reveal", {})
        if reveal:
            lines.append(
                f"    reveal: {reveal.get('reveal_k')}/{reveal.get('top_k')}"
                + ("  (HELD BACK)" if reveal.get("held_back") else ""))

        tr = turn.get("target_rank") or {}
        if tr:
            lines.append(
                f"    ▶ target rank: pool={tr.get('rank_in_pool')} "
                f"shown={tr.get('rank_shown')}")

        for note in turn.get("notes", []):
            lines.append(f"      · {note}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", help="path to trace.jsonl")
    parser.add_argument("--sample", help="render only this sample_id")
    parser.add_argument("--miss", action="store_true", help="only sessions where target was never shown")
    parser.add_argument("--limit", type=int, default=1, help="max sessions to render (0 = all)")
    args = parser.parse_args()

    sessions = load_sessions(args.trace)
    if args.sample:
        sessions = [s for s in sessions if s.get("sample_id") == args.sample]
    if args.miss:
        sessions = [s for s in sessions if not session_hit(s)]

    shown = sessions if args.limit == 0 else sessions[: args.limit]
    if not shown:
        print("(no matching sessions)")
        return
    print(f"# {len(sessions)} matching session(s); showing {len(shown)}\n")
    for session in shown:
        print(render(session))
        print()


if __name__ == "__main__":
    main()
