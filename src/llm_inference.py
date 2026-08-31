"""LLM-powered components: use-case inference, slot extraction, and response generation.

═══ OPTIONAL LAYER — NOT on the critical scored path. ═══
These are graceful-degradation hooks for unseen natural language. Every component fails safe to a
deterministic fallback when the LLM is unavailable, and all are token-metered via src/keys.py.
Their scored-metric contribution is currently unproven (slot extraction measured neutral); they
exist for robustness to language the static tables do not cover, not to win the public score.

Prompts live in prompts/ — edit them there, not here.
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
_DEFAULT_MODEL = "gemini-flash-lite-latest"


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

# The fixed output vocabulary. Passed to the model as a JSON schema enum so it can only ever
# emit one of these slots — novel language is MAPPED onto the fixed contract, never invented.
_VALID_SLOTS = [
    "material", "color", "size", "style", "use_case",
    "budget", "feature", "category", "brand",
]
_VALID_OPERATIONS = ["SET", "ADD", "REMOVE", "CLEAR", "NO_PREFERENCE"]
_SLOT_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "slot": {"type": "string", "enum": _VALID_SLOTS},
            "value": {"type": "string"},
            "polarity": {"type": "integer", "enum": [-1, 0, 1]},
            "operation": {"type": "string", "enum": _VALID_OPERATIONS},
        },
        "required": ["slot", "value", "polarity", "operation"],
    },
}


class LLMSlotExtractor:
    """Context-aware structured slot extraction from natural language using an LLM.

    Fills the gap where the regex SlotFiller has no vocabulary coverage ("budget-friendly",
    "my daughter's recital", "keeps me warm without the itch"). The conversation history,
    already-known slots, and product category are passed in so ambiguous messages
    ("something warmer") resolve against context.

    Output is constrained by a JSON schema whose `slot` field is an enum, so the model can
    only return one of the nine valid slots. Cached by (message + context) hash; falls back
    to an empty list when the LLM is unavailable or parsing fails.
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

    def extract(
        self,
        message: str,
        conversation: list[str] | None = None,
        known_slots: dict[str, str] | None = None,
        category: str | None = None,
    ) -> list[dict]:
        """Return ordered preference updates from `message`, read in the
        context of the conversation so far, the slots already known, and the product category.

        Returns [] on failure — callers treat this as "no additional constraints found".
        """
        if not self.available or not message.strip():
            return []
        payload = self._build_payload(message, conversation, known_slots, category)
        key = hashlib.sha1(payload.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        result = self._call_llm(payload)
        self._cache[key] = result
        self._flush_cache()
        return result

    @staticmethod
    def _build_payload(
        message: str, conversation: list[str] | None,
        known_slots: dict[str, str] | None, category: str | None,
    ) -> str:
        """Compact JSON context block the model reads alongside the new message."""
        recent = [m for m in (conversation or [])[-4:] if m.strip()]
        return json.dumps({
            "conversation_so_far": recent,
            "already_known": known_slots or {},
            "product_category": category,
            "new_message": message,
        }, ensure_ascii=False)

    def _call_llm(self, payload: str) -> list[dict]:
        try:
            resp = self._pool.generate_content(
                model=self._model,
                contents=payload,
                config=self._pool.types.GenerateContentConfig(
                    system_instruction=_SLOT_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                    response_schema=_SLOT_RESPONSE_SCHEMA,
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
            valid_slots = set(_VALID_SLOTS)
            valid_operations = set(_VALID_OPERATIONS)
            out = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                slot = str(item.get("slot", "")).strip()
                value = str(item.get("value", "")).strip().lower()
                polarity = int(item.get("polarity", 1))
                operation = str(item.get("operation", "")).strip().upper()
                # Backwards compatibility for entries created before the ledger schema.
                if not operation:
                    operation = "REMOVE" if polarity < 0 else "SET"
                value_ok = bool(value) or operation in {"CLEAR", "NO_PREFERENCE"}
                polarity_ok = polarity in (1, -1) or (
                    polarity == 0 and operation in {"CLEAR", "NO_PREFERENCE"})
                if (slot in valid_slots and operation in valid_operations
                        and value_ok and polarity_ok):
                    out.append({
                        "slot": slot, "value": value, "polarity": polarity,
                        "operation": operation,
                    })
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
