"""Lightweight, opt-in execution tracing for the shopping agent.

Purpose: capture *what the agent did and why* on every turn -- intent routing,
constraint/slot state, retrieval strategy and weights, ranking/coverage scores,
belief/convergence, clarification choice, the reveal decision, the final
response, and (for debugging misses) the ground-truth target's rank each turn.

Design goals:
    * ZERO behavioural impact. Tracing never influences a recommendation; the
      only target-aware field (target rank) is read-only debug annotation.
    * ZERO overhead when disabled. Every method short-circuits on `enabled`,
      so the official evaluator path (which never enables tracing) is unchanged.

Enable via environment:
    AGENT_TRACE=1               turn tracing on
    AGENT_TRACE_DIR=traces      output directory (default: traces/)

Output: one JSON object per session, appended to <dir>/trace.jsonl:
    {session_id, sample_id?, scenario_type?, difficulty_bucket?, category_bucket?,
     ground_truth?, user_profile?, turns: [ {turn, user_message, notes:[...],
     stages: {intent, constraints, retrieval, ranking, belief, clarification,
     reveal, response}, target_rank} ]}

Render a session for humans with scripts/show_trace.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


class Tracer:
    """Accumulates structured per-turn records and streams sessions to JSONL.

    Lifecycle (driven by the agent / eval runner):
        begin_session(session_id, meta) -> begin_turn(t, msg)
            -> stage(name, **fields) / note(...) (many times)
            -> end_turn() ... -> close()
    """

    def __init__(self, enabled: bool = False, out_dir: str | Path | None = None) -> None:
        self.enabled = enabled
        self.out_dir = Path(out_dir) if out_dir else None
        self._session: dict[str, Any] | None = None
        self._turn: dict[str, Any] | None = None
        self._fh = None
        self.sessions_written = 0
        # In-memory record of every completed session (used by the live UI /
        # programmatic callers; independent of file output).
        self.sessions: list[dict[str, Any]] = []

    # -- session lifecycle -------------------------------------------------
    def begin_session(self, session_id: str, meta: dict | None = None) -> None:
        if not self.enabled:
            return
        self._flush_session()
        self._session = {"session_id": session_id}
        if meta:
            self._session.update(meta)
        self._session["turns"] = []

    def target_asin(self) -> str | None:
        """Ground-truth ASIN for the active session, if the runner supplied it.

        Read-only: used solely to annotate the target's rank for debugging.
        """
        if not self.enabled or self._session is None:
            return None
        value = self._session.get("ground_truth")
        return str(value) if value else None

    # -- turn lifecycle ----------------------------------------------------
    def begin_turn(self, turn: int, user_message: str) -> None:
        if not self.enabled:
            return
        self._turn = {"turn": turn, "user_message": user_message, "notes": [], "stages": {}}

    def stage(self, name: str, **fields: Any) -> None:
        """Record (or merge) structured fields for a named pipeline stage."""
        if not self.enabled or self._turn is None:
            return
        self._turn["stages"].setdefault(name, {}).update(fields)

    def note(self, message: str) -> None:
        """Append a human-readable reasoning line to the current turn."""
        if not self.enabled or self._turn is None:
            return
        self._turn["notes"].append(message)

    def set_turn_field(self, **fields: Any) -> None:
        if not self.enabled or self._turn is None:
            return
        self._turn.update(fields)

    def end_turn(self) -> None:
        if not self.enabled or self._turn is None or self._session is None:
            return
        self._session["turns"].append(self._turn)
        self._turn = None

    # -- flushing ----------------------------------------------------------
    def _open(self) -> None:
        if self.out_dir is not None and self._fh is None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._fh = (self.out_dir / "trace.jsonl").open("w", encoding="utf-8")

    def _flush_session(self) -> None:
        if self._session is not None:
            self.sessions.append(self._session)
            if self.out_dir is not None:
                self._open()
                assert self._fh is not None
                self._fh.write(json.dumps(self._session) + "\n")
                self._fh.flush()
                self.sessions_written += 1
        self._session = None

    def close(self) -> None:
        if not self.enabled:
            return
        self._flush_session()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def get_tracer() -> Tracer:
    """Construct a Tracer from environment configuration (disabled by default)."""
    enabled = _truthy(os.getenv("AGENT_TRACE"))
    out_dir = os.getenv("AGENT_TRACE_DIR", "traces") if enabled else None
    return Tracer(enabled=enabled, out_dir=out_dir)
