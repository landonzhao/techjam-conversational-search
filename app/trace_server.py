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
