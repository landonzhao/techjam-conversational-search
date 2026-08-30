"""Conversation-aware rerankers (local cross-encoder + Gemini LLM).

═══ OPTIONAL LAYER — OFF by default, NOT on the critical scored path. ═══
Both rerankers are disabled in scoring (USE_CROSS_ENCODER / USE_LLM_RERANK = False): the local
cross-encoder measured neutral/negative, and the LLM reranker is rate-limited. They are wired,
gated on near-ties, and token-metered so they can be enabled and measured — see the flag ledger in
src/agent.py. This is the "LLM Semantic Ranking" pillar hook; keep only on a measured win.

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

DEFAULT_MODEL = "gemini-flash-lite-latest"

MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|alloy|"
    r"stainless steel|sterling silver|silver|gold|brass|denim|linen|suede|rubber)\b",
    re.I,
)

SYSTEM_PROMPT = (
    "You are an expert personal shopper for a clothing, shoes and jewellery store. The "
    "customer is looking for ONE specific product. You are given their stated constraints "
    "and a numbered list of candidate products with their attributes.\n\n"
    "Choose the best match, paying attention to the attributes that DIFFERENTIATE "
    "otherwise-similar products (model/version, pattern, silhouette, cut, distinctive "
    "features) — these matter more than attributes all candidates share. Hard constraints "
    "(material, product type, explicit features) must be satisfied; a wrong product type or "
    "wrong material disqualifies a candidate. Do not reward popularity or brand fame — judge "
    "only fit to the customer's description.\n\n"
    'Return ONLY JSON: {"order": [best..worst candidate numbers], "why": "<=15 words on the '
    'top pick"}. Every candidate number must appear exactly once in "order". No prose.'
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
        """Listwise rerank: the LLM returns candidates ordered best-to-worst; we apply that
        permutation to the head and keep the tail as-is. Fail-safe to input order."""
        if not self.available or not asins:
            return asins
        head = asins[:depth]
        constraints = self._clean_constraints(conversation)
        listing = "\n".join(self._candidate_line(i, a) for i, a in enumerate(head))
        prompt = (
            f"Customer's constraints (most recent first):\n{constraints}\n\n"
            f"Candidates:\n{listing}\n\n"
            f'Order all candidates 0-{len(head)-1} best to worst.'
        )
        try:
            resp = self._pool.generate_content(
                model=self.model,
                contents=prompt,
                config=self._pool.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                ),
            )
            order = self._parse_order(resp.text, len(head))
        except Exception:
            return asins  # fail safe
        if not order:
            return asins
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
    def _parse_order(text: str, n: int) -> list[int]:
        """Parse the listwise {"order": [...]} permutation. Defensive: dedupes, drops
        out-of-range indices, and appends any candidates the model omitted (in original
        order) so the result is always a full valid permutation of 0..n-1."""
        if not text:
            return []
        raw: list = []
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("order"), list):
                raw = data["order"]
            elif isinstance(data, list):
                raw = data
        except Exception:
            m = re.search(r'\[([0-9,\s]+)\]', text)
            if m:
                raw = [p for p in m.group(1).split(",")]
        seen: set[int] = set()
        order: list[int] = []
        for x in raw:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= i < n and i not in seen:
                seen.add(i)
                order.append(i)
        if not order:
            return []
        order.extend(i for i in range(n) if i not in seen)  # append omitted, original order
        return order
