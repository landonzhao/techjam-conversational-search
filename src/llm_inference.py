"""LLM-powered components: use-case inference, slot extraction, and response generation.

Prompts live in prompts/ — edit them there, not here.
All components fail safe to their deterministic fallback when the LLM is unavailable.
"""
from __future__ import annotations

import hashlib
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


# ---------------------------------------------------------------------------

_SLOT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "slot_extraction.txt"
_SLOT_SYSTEM_PROMPT: str = (
    _SLOT_PROMPT_PATH.read_text(encoding="utf-8") if _SLOT_PROMPT_PATH.exists() else ""
)
_SLOT_CACHE_PATH = Path("cache/llm_slot_cache.json")


class LLMSlotExtractor:
    """Extract structured slot constraints from natural language using an LLM.

    Fills the gap where the regex SlotFiller has no vocabulary coverage:
    "budget-friendly", "office appropriate", "easy to wash", "not too flashy", etc.

    Responses are cached by message hash so each unique phrasing is only called once.
    Falls back to an empty list when the LLM is unavailable or parsing fails.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model
        self._pool = GeminiClientPool()
        self._cache: dict[str, list[dict]] = {}
        if _SLOT_CACHE_PATH.exists():
            try:
                self._cache = json.loads(_SLOT_CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    @property
    def available(self) -> bool:
        return self._pool.available and bool(_SLOT_SYSTEM_PROMPT)

    def extract(self, message: str) -> list[dict]:
        """Return a list of {slot, value, polarity} dicts extracted from the message.

        Returns [] on failure — callers should treat this as "no additional constraints found".
        """
        if not self.available or not message.strip():
            return []
        key = hashlib.sha1(message.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        result = self._call_llm(message)
        self._cache[key] = result
        self._flush_cache()
        return result

    def _call_llm(self, message: str) -> list[dict]:
        try:
            resp = self._pool.generate_content(
                model=self._model,
                contents=message,
                config=self._pool.types.GenerateContentConfig(
                    system_instruction=_SLOT_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                ),
            )
            return self._parse(resp.text)
        except Exception:
            return []

    @staticmethod
    def _parse(text: str) -> list[dict]:
        if not text:
            return []
        try:
            data = json.loads(text)
            if not isinstance(data, list):
                return []
            valid_slots = {
                "material", "color", "size", "style", "use_case",
                "budget", "feature", "category", "brand",
            }
            out = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                slot = str(item.get("slot", "")).strip()
                value = str(item.get("value", "")).strip().lower()
                polarity = int(item.get("polarity", 1))
                if slot in valid_slots and value and polarity in (1, -1):
                    out.append({"slot": slot, "value": value, "polarity": polarity})
            return out
        except Exception:
            return []

    def _flush_cache(self) -> None:
        try:
            _SLOT_CACHE_PATH.parent.mkdir(exist_ok=True)
            _SLOT_CACHE_PATH.write_text(
                json.dumps(self._cache, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------

_RESPONSE_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "response_generation.txt"
_RESPONSE_SYSTEM_PROMPT: str = (
    _RESPONSE_PROMPT_PATH.read_text(encoding="utf-8") if _RESPONSE_PROMPT_PATH.exists() else ""
)


class LLMResponseGenerator:
    """Generate natural conversational responses instead of slot-fill templates.

    Takes the conversation context, top candidate titles, and the next clarifying
    question, and returns a single natural sentence. Falls back to the template
    message if the LLM is unavailable.

    Responses are not cached — each turn is unique.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model
        self._pool = GeminiClientPool()

    @property
    def available(self) -> bool:
        return self._pool.available and bool(_RESPONSE_SYSTEM_PROMPT)

    def generate(
        self,
        conversation: list[str],
        top_titles: list[str],
        ask_slot: str | None,
        ask_phrasing: str,
        constraints_found: list[str],
        fallback: str,
    ) -> str:
        """Return a natural response, or fallback if the LLM is unavailable."""
        if not self.available:
            return fallback
        payload = json.dumps({
            "conversation": conversation[-3:],
            "top_products": top_titles[:2],
            "ask_slot": ask_slot,
            "ask_phrasing": ask_phrasing,
            "constraints_found": constraints_found,
        })
        try:
            resp = self._pool.generate_content(
                model=self._model,
                contents=payload,
                config=self._pool.types.GenerateContentConfig(
                    system_instruction=_RESPONSE_SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=100,
                ),
            )
            text = (resp.text or "").strip()
            return text if text else fallback
        except Exception:
            return fallback
