"""Phase 4 — conversation-aware LLM reranker (Gemini).

Reorders the retrieved candidate pool against the full conversation so the exact
target rises toward rank 1. Provider-agnostic in spirit; this implementation uses
Google Gemini. Key is read from the environment (GEMINI_API_KEY / GOOGLE_API_KEY),
never hard-coded.

Fails safe: if the SDK is missing, no key is set, or any call errors/times out,
`rerank()` returns the input order unchanged, so the agent stays offline-safe and
never crashes a scoring run.
"""
from __future__ import annotations

import json
import re

from src.keys import GeminiClientPool

DEFAULT_MODEL = "gemini-2.5-flash-lite"

MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|alloy|"
    r"stainless steel|sterling silver|silver|gold|brass|denim|linen|suede|rubber)\b",
    re.I,
)

SYSTEM_PROMPT = (
    "You are matching a shopper to the ONE specific product they are describing in a "
    "clothing, shoes and jewelry store. You are given the shopper's stated constraints "
    "and a numbered list of candidate products with their attributes. Score how well "
    "EACH candidate matches the shopper's hard constraints (material, type, distinctive "
    "features); soft preferences break ties. Reward exact attribute matches; penalize a "
    "wrong product type or wrong material. Return ONLY a JSON object mapping candidate "
    'number to a 0-100 match score, e.g. {"0": 95, "1": 40}. No prose.'
)


class CrossEncoderReranker:
    """Local semantic reranker (cross-encoder/ms-marco-MiniLM-L-6-v2).

    Free, unlimited, offline. Scores (constraints, candidate-text) pairs and returns
    the head reordered by score. Loaded lazily; if unavailable, `available` is False
    and the Agent skips it. No API key, no rate limits.
    """

    def __init__(self, catalog: dict[str, dict],
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.catalog = catalog
        self.model = None
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _doc(self, asin: str) -> str:
        p = self.catalog.get(asin, {})
        title = str(p.get("title") or "")
        cats = " ".join(str(c) for c in (p.get("categories") or [])[-3:])
        feats = " ".join(str(f) for f in (p.get("features") or [])[:5])
        return f"{title} {cats} {feats}"[:500]

    def scores(self, query: str, asins: list[str], depth: int) -> list[float]:
        """Cross-encoder relevance score for each of the first `depth` candidates."""
        if not self.available or not asins or not query.strip():
            return []
        head = asins[:depth]
        pairs = [(query, self._doc(a)) for a in head]
        try:
            return [float(s) for s in self.model.predict(pairs, show_progress_bar=False)]
        except Exception:
            return []


class LLMReranker:
    def __init__(self, catalog: dict[str, dict], model: str = DEFAULT_MODEL) -> None:
        self.catalog = catalog
        self.model = model
        self._pool = GeminiClientPool()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def available(self) -> bool:
        return self._pool.available

    def _candidate_line(self, idx: int, asin: str) -> str:
        """Rich description: the attributes the shopper is actually constraining on."""
        p = self.catalog.get(asin, {})
        title = str(p.get("title") or "")[:80]
        cats = p.get("categories") or []
        cat = " > ".join(str(c) for c in cats[-2:]) if isinstance(cats, list) else ""
        feats = p.get("features") or []
        corpus = " ".join(str(f) for f in feats)
        mat = MATERIAL_RE.search(corpus + " " + title)
        material = mat.group(0).lower() if mat else "?"
        # two short, informative feature snippets (skip long marketing blobs)
        snips = [str(f)[:45] for f in feats if 3 < len(str(f)) < 60][:2]
        feat_s = "; ".join(snips)
        price = p.get("price")
        price_s = f"${price}" if price not in (None, "") else "?"
        return f"{idx}. {title} | type: {cat} | material: {material} | {feat_s} | {price_s}"

    def rerank(self, conversation: list[str], asins: list[str], top_k: int, depth: int = 20) -> list[str]:
        """Score-based rerank: LLM scores each candidate 0-100; we sort by score,
        breaking ties by the incoming (strong) order. Fail-safe to input order."""
        if not self.available or not asins:
            return asins
        head = asins[:depth]
        constraints = self._clean_constraints(conversation)
        listing = "\n".join(self._candidate_line(i, a) for i, a in enumerate(head))
        prompt = (
            f"Shopper's constraints (most recent first):\n{constraints}\n\n"
            f"Candidates:\n{listing}\n\n"
            f'Return {{"number": score}} for every candidate 0-{len(head)-1}.'
        )
        try:
            resp = self._pool.generate_content(
                model=self.model,
                contents=prompt,
                config=self._pool.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )
            scores = self._parse_scores(resp.text, len(head))
            usage = getattr(resp, "usage_metadata", None)
            if usage:
                self.prompt_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
                self.completion_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
        except Exception:
            return asins  # fail safe
        if not scores:
            return asins
        # sort head by (LLM score desc, original rank asc); keep tail as-is
        order = sorted(range(len(head)), key=lambda i: (-scores.get(i, -1.0), i))
        reordered = [head[i] for i in order] + asins[depth:]
        return reordered

    @staticmethod
    def _clean_constraints(conversation: list[str]) -> str:
        """Strip the simulator scaffolding, keep the actual constraint phrases."""
        lines = []
        for msg in conversation[-6:]:
            t = msg
            for junk in ("I'm looking for", "A key requirement is:",
                         "For that, what matters is:", "Actually, ignore my earlier preference.",
                         "What I need is:"):
                t = t.replace(junk, "")
            t = t.strip(" .:-")
            # drop long marketing blobs (keep concise constraint phrases)
            t = "; ".join(part.strip()[:60] for part in t.split(";") if 2 < len(part.strip()) < 80)
            if t:
                lines.append(f"- {t}")
        return "\n".join(reversed(lines)) or "- (no specific constraints yet)"

    @staticmethod
    def _parse_scores(text: str, n: int) -> dict[int, float]:
        if not text:
            return {}
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                out = {}
                for k, v in data.items():
                    if str(k).strip().lstrip("-").isdigit():
                        try:
                            out[int(k)] = float(v)
                        except (TypeError, ValueError):
                            pass
                return out
        except Exception:
            pass
        return {}
