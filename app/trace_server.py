"""Dev-only web UI to watch agent conversations and inspect the reasoning.

Two things in one page:
  1. A conversation panel that replays a user<->agent session turn by turn.
  2. An inspector: click any agent turn to see what happened behind the scenes
     (intent routing, retrieval strategy, ranking, belief, clarification, reveal,
     and where the ground-truth target sat in the ranking).

You can either LIVE-SIMULATE a sample (runs the real agent through the official
evaluator's conversation loop) or LOAD an existing traces/*/trace.jsonl file.

This is tooling, not part of the shipped agent. Run:
    python app/trace_server.py           # http://127.0.0.1:5001
    python app/trace_server.py --port 8000

Requires Flask (already a dev dependency).
"""
from __future__ import annotations

import argparse
import glob
import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from src.agent import Agent  # noqa: E402
from src.trace import Tracer  # noqa: E402

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))

# Lazily-initialised, cached singletons (the agent loads models once).
_CATALOG_PATH = str(ROOT / "data" / "catalog.jsonl")
_catalog_index = None
_agent = None
_chat_sessions: dict[str, dict] = {}


class TracingAgent(Agent):
    """Agent that tags each session with its sample metadata for tracing."""

    def bind_samples(self, samples: list[dict]) -> None:
        self._trace_samples = list(samples)
        self._trace_cursor = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        samples = getattr(self, "_trace_samples", None)
        if samples and self._trace_cursor < len(samples):
            sample = samples[self._trace_cursor]
            self._trace_cursor += 1
            self._pending_meta = {
                "sample_id": sample.get("sample_id"),
                "scenario_type": sample.get("scenario_type"),
                "difficulty_bucket": sample.get("difficulty_bucket"),
                "category_bucket": sample.get("category_bucket"),
                "ground_truth": sample.get("ground_truth", {}).get("parent_asin"),
            }
        super().reset(session_id, user_profile)


def get_catalog_index():
    global _catalog_index
    if _catalog_index is None:
        _catalog_index = catalog_index(_CATALOG_PATH)
    return _catalog_index


def get_agent() -> TracingAgent:
    global _agent
    if _agent is None:
        _agent = TracingAgent(_CATALOG_PATH)
    return _agent


def _dataset_paths() -> list[Path]:
    paths: list[Path] = []
    for pattern in ("data/*.jsonl", "data/test_suite/*.jsonl"):
        for p in sorted(glob.glob(str(ROOT / pattern))):
            path = Path(p)
            if path.name == "catalog.jsonl":
                continue
            paths.append(path)
    return paths


def _safe_path(raw: str) -> Path:
    """Resolve a user-supplied path and confine it to the repo."""
    path = (ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if ROOT not in path.parents and path != ROOT:
        raise ValueError("path escapes repository")
    return path


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/datasets")
def datasets():
    out = []
    for path in _dataset_paths():
        try:
            count = sum(1 for _ in path.open())
        except OSError:
            count = 0
        out.append({"path": str(path.relative_to(ROOT)), "name": path.stem, "count": count})
    return jsonify(out)


@app.get("/api/traces")
def traces():
    out = []
    for p in sorted(glob.glob(str(ROOT / "traces" / "**" / "trace.jsonl"), recursive=True)):
        path = Path(p)
        out.append({"path": str(path.relative_to(ROOT)), "name": str(path.parent.name)})
    return jsonify(out)


@app.get("/api/samples")
def samples():
    path = _safe_path(request.args.get("path", ""))
    rows = load_jsonl(path)
    out = [{
        "index": i,
        "sample_id": r.get("sample_id"),
        "scenario_type": r.get("scenario_type"),
        "difficulty_bucket": r.get("difficulty_bucket"),
        "category_bucket": r.get("category_bucket"),
        "ground_truth": (r.get("ground_truth") or {}).get("parent_asin"),
    } for i, r in enumerate(rows[:500])]
    return jsonify(out)


@app.get("/api/trace")
def trace_file():
    path = _safe_path(request.args.get("path", ""))
    sessions = load_jsonl(path)[:200]
    return jsonify(sessions)


def _chat_snapshot(agent: Agent, session_id: str, turn: dict | None = None) -> dict:
    """Return safe, presentation-oriented state for the manual demo chat.

    This is intentionally a read-only projection. It exposes decisions needed to explain the
    system (active/retired ledger, query, route, belief, and action) without exposing a target ASIN
    or changing the production/evaluator response contract.
    """
    state = agent._sessions.get(session_id)
    if state is None:
        return {}
    return {
        "intent": state.intent,
        "buying_score": round(state.buying_score, 4),
        "phase": state.phase,
        "conv_state": state.conv_state,
        "need": state.need.describe(),
        "active_slots": {c.slot: c.value for c in state.need.positives()},
        "active_ledger": [{
            "operation": c.operation, "slot": c.slot, "value": c.value,
            "polarity": c.polarity, "turn": c.turn,
        } for c in state.need.ledger if c.active and c.polarity > 0 and c.value],
        "retired_ledger": [{
            "operation": c.operation, "slot": c.slot, "value": c.value,
            "polarity": c.polarity, "turn": c.turn,
        } for c in state.need.ledger if (not c.active or c.polarity <= 0) and c.value],
        "excluded_terms": sorted(state.need.excluded_terms()),
        "no_preference": sorted(state.need.no_preference),
        "effective_query": state.query_text(),
        "retrieval_query": state.retrieval_query(),
        "leaky_evidence": bool(state.leaky_evidence),
        "leaky_ranking_evidence": bool(state.leaky_ranking_evidence),
        "pool_size": state.last_pool,
        "belief": {
            "confidence": round(state.belief.confidence, 4),
            "item_confidence": round(state.belief.item_confidence, 4),
            "margin": round(state.belief.margin, 4),
            "entropy": round(state.belief.entropy, 4),
            "uncertainty": {k: round(v, 4) for k, v in state.belief.attr_uncertainty.items()},
        },
        "ask_attribute": (turn or {}).get("stages", {}).get("response", {}).get("ask_attribute")
        if turn else state.ig_attr,
    }


@app.post("/api/chat/reset")
def chat_reset():
    """Start a clean, non-target-aware manual demo session."""
    body = request.get_json(silent=True) or {}
    session_id = f"demo_{uuid.uuid4().hex}"
    agent = get_agent()
    # The cached tracing agent is also used by live simulation. Clear its sample cursor and start a
    # fresh in-memory tracer so manual chat never inherits evaluator metadata or a ground truth.
    agent._trace_samples = []
    agent._trace_cursor = 0
    agent._pending_meta = {}
    agent._tracer = Tracer(enabled=True)
    profile = {
        "preference_tags": [str(x).strip() for x in (body.get("preference_tags") or []) if str(x).strip()],
        "summary": str(body.get("summary") or ""),
    }
    agent.reset(session_id, profile)
    _chat_sessions[session_id] = {"turn": 0, "profile": profile, "agent": agent}
    return jsonify({"session_id": session_id, "turn": 0, "profile": profile, "state": _chat_snapshot(agent, session_id)})


@app.post("/api/chat/respond")
def chat_respond():
    """Run one real Agent turn and return its response plus the presentation trace."""
    body = request.get_json(force=True)
    session_id = str(body.get("session_id") or "")
    message = str(body.get("message") or "").strip()
    session = _chat_sessions.get(session_id)
    if not session:
        return jsonify({"error": "chat session not found; start a new chat"}), 404
    if not message:
        return jsonify({"error": "message is empty"}), 400
    if session["turn"] >= 10:
        return jsonify({"error": "session reached the 10-turn limit"}), 400
    agent = session["agent"]
    session["turn"] += 1
    turn_number = session["turn"]
    response = agent.respond(session_id, message, turn_number, 10)
    trace = None
    tracer_session = getattr(agent._tracer, "_session", None)
    if tracer_session and tracer_session.get("turns"):
        trace = tracer_session["turns"][-1]
    return jsonify({
        "session_id": session_id,
        "turn": turn_number,
        "user_message": message,
        "response": response,
        "state": _chat_snapshot(agent, session_id, trace),
        "trace": trace or {},
    })


@app.post("/api/simulate")
def simulate():
    body = request.get_json(force=True)
    dataset = _safe_path(body["path"])
    index = int(body["index"])
    rows = load_jsonl(dataset)
    if not 0 <= index < len(rows):
        return jsonify({"error": "index out of range"}), 400
    sample = rows[index]

    catalog_ids, categories, products = get_catalog_index()
    agent = get_agent()
    agent._tracer = Tracer(enabled=True)  # fresh, in-memory
    agent.bind_samples([sample])
    evaluate(agent, [sample], catalog_ids, categories, products)
    agent._tracer.close()

    if not agent._tracer.sessions:
        return jsonify({"error": "no trace produced"}), 500
    return jsonify(agent._tracer.sessions[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    print(f"Trace UI → http://{args.host}:{args.port}")
    # threaded=False: the cached agent holds per-session state; serialise requests.
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
