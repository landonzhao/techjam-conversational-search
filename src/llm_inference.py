"""LLM-powered inference for use-case → implied product attributes.

- System prompt lives in prompts/use_case_inference.txt — edit it there, not here.
- Responses are cached to cache/llm_inference_cache.json keyed on (use_cases, category).
  The LLM is called only once per unique combination; all later calls hit the cache.
- Falls back to USE_CASE_LEXICON when no key is configured, the call fails, or parsing fails.
- Feature-flagged: only active when Agent.USE_LLM_INFERENCE = True.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.keys import GeminiClientPool
from src.understanding import USE_CASE_LEXICON, NeedModel

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "use_case_inference.txt"
_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8") if _PROMPT_PATH.exists() else ""

_CACHE_PATH = Path("cache/llm_inference_cache.json")
_DEFAULT_MODEL = "gemini-2.5-flash-lite"


class LLMUseCaseInferrer:
    """Infer implied product attributes from a shopper's stated occasions.

    Falls back to the deterministic USE_CASE_LEXICON when the LLM is unavailable.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model
        self._pool = GeminiClientPool()
        self._cache: dict[str, list[str]] = {}

        if _CACHE_PATH.exists():
            try:
                self._cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    @property
    def available(self) -> bool:
        return self._pool.available and bool(_SYSTEM_PROMPT)

    def infer(self, use_cases: set[str], category: str | None = None) -> set[str]:
        """Return implied attribute terms for the given occasions + category."""
        if not use_cases:
            return set()

        static_terms: set[str] = set()
        for uc in use_cases:
            entry = USE_CASE_LEXICON.get(uc)
            if entry:
                static_terms |= entry.get("terms", set())

        if not self.available:
            return static_terms

        cache_key = self._key(use_cases, category)
        if cache_key in self._cache:
            llm_terms = set(self._cache[cache_key])
        else:
            llm_terms = self._call_llm(use_cases, category)
            if llm_terms is not None:
                self._cache[cache_key] = sorted(llm_terms)
                self._flush_cache()

        return static_terms | (llm_terms or set())

    def _key(self, use_cases: set[str], category: str | None) -> str:
        return json.dumps({"use_cases": sorted(use_cases), "category": category}, sort_keys=True)

    def _call_llm(self, use_cases: set[str], category: str | None) -> set[str] | None:
        payload = json.dumps({"use_cases": sorted(use_cases), "category": category})
        try:
            resp = self._pool.generate_content(
                model=self._model,
                contents=payload,
                config=self._pool.types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=256,
                    response_mime_type="application/json",
                ),
            )
            return self._parse(resp.text)
        except Exception:
            return None

    @staticmethod
    def _parse(text: str) -> set[str] | None:
        if not text:
            return None
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "terms" in data:
                terms = data["terms"]
                if isinstance(terms, list):
                    return {str(t).lower().strip() for t in terms if t}
        except Exception:
            m = re.search(r'\[([^\]]+)\]', text)
            if m:
                try:
                    items = json.loads(f"[{m.group(1)}]")
                    return {str(t).lower().strip() for t in items if t}
                except Exception:
                    pass
        return None

    def _flush_cache(self) -> None:
        try:
            _CACHE_PATH.parent.mkdir(exist_ok=True)
            _CACHE_PATH.write_text(
                json.dumps(self._cache, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass


class SmartUseCaseInferencer:
    """Drop-in replacement for UseCaseInferencer that uses the LLM when available.

    Returns the same dict shape: {"cases": set, "terms": set}
    """

    def __init__(self) -> None:
        self._llm = LLMUseCaseInferrer()

    @property
    def llm_available(self) -> bool:
        return self._llm.available

    def infer(self, need: NeedModel) -> dict:
        cases = {c.value for c in need.positives("use_case")}
        terms = self._llm.infer(cases, need.category)
        return {"cases": cases, "terms": terms}
