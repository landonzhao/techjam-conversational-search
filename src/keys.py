"""Gemini API key pool with round-robin rotation and rate-limit retry.

Reads GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ... from the environment.
On a 429 / quota / rate-limit response the pool silently rotates to the next key.
"""
from __future__ import annotations

import itertools
import os


def _load_gemini_keys() -> list[str]:
    keys: list[str] = []
    base = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if base:
        keys.append(base)
    for i in range(2, 32):
        k = os.environ.get(f"GEMINI_API_KEY_{i}")
        if not k:
            break
        keys.append(k)
    return keys


class GeminiClientPool:
    """Round-robin pool of google.genai clients.

    Usage mirrors a single client:
        pool = GeminiClientPool()
        if pool.available:
            resp = pool.generate_content(model=..., contents=..., config=...)

    Token metering is process-wide: every successful call — from any component and any
    pool instance — increments the class-level counters below. This is the single choke
    point through which all Gemini traffic flows, so the agent can report honest per-turn
    token usage by snapshotting `usage_totals()` around each turn, regardless of which
    component (slot extraction, use-case inference, reranker, response generation) made
    the call. Cached calls make no API request and correctly cost zero.
    """

    _total_prompt_tokens = 0
    _total_completion_tokens = 0

    @classmethod
    def usage_totals(cls) -> tuple[int, int]:
        """Cumulative (prompt_tokens, completion_tokens) across every Gemini call so far."""
        return cls._total_prompt_tokens, cls._total_completion_tokens

    @classmethod
    def _record_usage(cls, resp: object) -> None:
        usage = getattr(resp, "usage_metadata", None)
        if usage is None:
            return
        cls._total_prompt_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
        cls._total_completion_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)

    def __init__(self) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
            self._genai = genai
            self.types = genai_types
        except Exception:
            self._genai = None
            self.types = None
            self._clients: list = []
            self._cycle = iter([])
            return

        self._clients = []
        for key in _load_gemini_keys():
            try:
                self._clients.append(self._genai.Client(api_key=key))
            except Exception:
                pass
        self._cycle = itertools.cycle(self._clients) if self._clients else iter([])

    @property
    def available(self) -> bool:
        return bool(self._clients)

    def generate_content(self, model: str, contents, config) -> object:
        """Call generate_content, rotating to the next key on quota/rate-limit errors."""
        if not self._clients:
            raise RuntimeError("No Gemini API keys configured")
        last_exc: Exception | None = None
        for _ in range(len(self._clients)):
            client = next(self._cycle)
            try:
                resp = client.models.generate_content(
                    model=model, contents=contents, config=config)
                self._record_usage(resp)
                return resp
            except Exception as exc:
                msg = str(exc).lower()
                if any(t in msg for t in ("429", "quota", "rate limit", "resource_exhausted")):
                    last_exc = exc
                    continue
                raise
        raise last_exc  # type: ignore[misc]
