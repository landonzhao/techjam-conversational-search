"""Catalog loading, text normalization, and BM25 retrieval.

This module owns the raw data layer:
  - text normalization utilities shared by all components
  - loading catalog.jsonl into memory and an FTS5 index
  - BM25 search against that index

Everything downstream (VectorRetriever, CoverageReranker, Personalizer, …) receives
the `catalog` dict directly — no module imports this except agent.py and tests.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from src.config import BM25_MAX_TERMS, BM25_WEIGHTS

# ---------------------------------------------------------------------------
# Shared text primitives (single definition; understanding.py imports from here)

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    # conversational scaffolding from the simulator — noise, not product signal
    "actually", "ignore", "earlier", "preference", "what", "matters", "dont",
    "have", "additional", "prefer", "use", "your", "judgment", "need", "key",
    "requirement", "still", "exploring", "not", "sure",
}


def text(value: object) -> str:
    """Flatten any catalog field (dict/list/scalar) to a plain string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(raw: str) -> list[str]:
    """Tokenise, lowercase, drop stopwords and single-char tokens."""
    return [
        t.lower()
        for t in TOKEN_RE.findall(raw)
        if len(t) > 1 and t.lower() not in STOPWORDS
    ]


# ---------------------------------------------------------------------------
# Catalog

class Catalog:
    """In-memory catalog dict + SQLite FTS5 index for BM25 retrieval.

    Build once at agent startup; shared (read-only) across all sessions.
    """

    def __init__(self, catalog_path: str | Path) -> None:
        self.path = Path(catalog_path)
        self.products: dict[str, dict] = {}
        self._conn = sqlite3.connect(":memory:")
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                p = json.loads(line)
                asin = str(p["parent_asin"])
                self.products[asin] = p
                batch.append((
                    asin,
                    text(p.get("title")),
                    text(p.get("categories")),
                    text(p.get("features")),
                    text(p.get("details")),
                    text(p.get("store")),
                    text(p.get("description")),
                ))
                if len(batch) >= 1000:
                    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self._conn.commit()

    # ------------------------------------------------------------------ BM25
    def bm25(self, query: str, pool: int) -> list[str]:
        """FTS5 BM25 search. Returns up to `pool` ASINs ranked by BM25 score."""
        unique = list(dict.fromkeys(terms(query)))[:BM25_MAX_TERMS]
        if not unique:
            return []
        expr = " OR ".join(f'"{t}"' for t in unique)
        w = BM25_WEIGHTS
        rows = self._conn.execute(
            f"SELECT parent_asin FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, {w[0]}, {w[1]}, {w[2]}, {w[3]}, {w[4]}, {w[5]}, {w[6]}) LIMIT ?",
            (expr, pool),
        ).fetchall()
        return [str(r[0]) for r in rows]

    # ------------------------------------------------------------------ helpers
    def get(self, asin: str, default: dict | None = None) -> dict:
        return self.products.get(asin, default or {})

    def __contains__(self, asin: str) -> bool:
        return asin in self.products

    def __len__(self) -> int:
        return len(self.products)
