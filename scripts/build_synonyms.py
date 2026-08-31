"""Build a data-driven synonym/expansion table from catalog embeddings.

How it works
------------
For each "seed" term (a word shoppers use), we:
  1. Encode the seed as a semantic query using BGE (the same model used for retrieval)
  2. Find the top-K most semantically similar products in the catalog
  3. Extract vocabulary that is *distinctive* in those products vs the full catalog
     (high relative frequency = terms that describe the same concept)

This finds what the catalog actually says about a concept, even when the shopper
uses different words. For example:
  - "warm" → products about warmth say: insulated, thermal, fleece, down, sherpa
  - "vegan" → products for vegan shoppers say: faux, synthetic, pu leather, manmade
  - "compression" → products in compression category say: moisture-wicking, yoga, athletic

The output is merged with the hand-written EXPANSIONS table so both cover different
vocabulary — the hand-written table handles unambiguous synonyms (grey↔gray), the
data-driven table handles catalog-specific vocabulary a human couldn't anticipate.

Output
------
cache/synonyms.json  — loaded automatically by ExpansionTable.load()

Usage
-----
  python scripts/build_synonyms.py                          # all seeds from SEEDS list
  python scripts/build_synonyms.py --seeds "vegan,merino"  # specific seeds only
  python scripts/build_synonyms.py --top-k 60 --top-n 12   # more expansions
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.catalog import Catalog, TOKEN_RE, text
from src.config import EMBED_MODEL, EMBED_QUERY_PREFIX

# ---------------------------------------------------------------------------
# Seed terms: concepts shoppers say that may not match catalog vocabulary.
# Organised by gap type so it's easy to add more.
SEEDS: list[str] = [
    # Material alternatives / non-obvious catalog vocabulary
    "vegan", "faux leather", "synthetic leather",
    "merino", "cashmere", "bamboo", "linen", "modal", "recycled",
    "sustainable", "eco friendly", "organic",
    "sherpa", "fleece", "thermal", "ribbed", "woven", "knit",
    # Attribute synonyms
    "warm", "cozy", "lightweight", "breathable", "stretchy", "durable",
    "waterproof", "water resistant", "windproof",
    "compression", "supportive", "padded", "quilted", "lined",
    "anti odor", "moisture wicking", "quick dry",
    "uv protection", "reflective",
    # Style descriptors
    "oversized", "relaxed", "fitted", "slim", "boxy", "cropped",
    "vintage", "retro", "minimalist", "streetwear", "athleisure",
    "boho", "preppy", "classic",
    # Functional features
    "pockets", "adjustable", "packable", "reversible",
    "high waist", "plus size",
    # Occasions the USE_CASE_LEXICON may miss
    "yoga", "pilates", "cycling", "climbing", "skiing", "snowboarding",
    "business casual", "cocktail", "brunch", "date night",
    "beach", "pool", "resort", "festival",
    "maternity", "nursing", "postpartum",
    # Care / construction that shoppers mention but listings use different terms
    "machine washable", "hand wash", "non iron", "wrinkle free",
    "non slip", "slip resistant", "steel toe", "safety",
]

# Terms that should NEVER appear as expansion output — too generic or misleading
BLACKLIST = {
    # Functional stopwords
    "the", "and", "for", "with", "this", "that", "from", "are", "was",
    "our", "your", "its", "not", "can", "has", "have", "will", "may",
    "all", "any", "each", "per",
    # Shopping filler
    "product", "item", "style", "color", "size", "great", "best", "good",
    "quality", "perfect", "nice", "love", "new", "top", "high", "easy",
    "free", "get", "use", "made", "make", "like", "fit", "look", "feel",
    "wear", "take", "give", "keep",
    # Measurements / units
    "cm", "mm", "inch", "oz", "lb", "pack", "pair", "pieces", "piece",
    "set", "count", "lot",
    # Demographics (too broad)
    "women", "men", "man", "woman", "boys", "girls", "kids", "unisex",
    "adult", "junior", "youth",
    # Numeric size tokens
    "xs", "xl", "xxl", "xxxl",
    # Colour names (expansion should be about attributes, not colours)
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "navy", "beige", "yellow", "purple", "orange", "gold",
    # Generic description noise
    "about", "due", "below", "above", "note", "please", "carefully",
    "attention", "ordering", "approximately", "measures",
}

MIN_TOKEN_LEN = 3


def _tokens(s: str) -> list[str]:
    return [
        t.lower() for t in TOKEN_RE.findall(s)
        if len(t) >= MIN_TOKEN_LEN and t.lower() not in BLACKLIST
    ]


def _product_text(p: dict) -> str:
    return " ".join([
        text(p.get("title", "")),
        text(p.get("features", "")),
    ])


def build(
    catalog_path: str = "data/catalog.jsonl",
    emb_path: str = "cache/embeddings.npy",
    asins_path: str = "cache/asins.json",
    out_path: str = "cache/synonyms.json",
    top_k: int = 50,       # nearest-neighbour products to inspect per seed
    top_n: int = 12,       # max expansion terms to keep per seed
    min_lift: float = 2.5, # expansion term must be ≥ N× more common in neighbours than catalog
    min_freq: float = 0.12, # must appear in ≥ this fraction of the top-K neighbours
    seeds: list[str] | None = None,
) -> dict[str, list[str]]:
    from sentence_transformers import SentenceTransformer

    seed_list = seeds if seeds is not None else SEEDS

    print("Loading catalog …")
    cat = Catalog(catalog_path)
    asin_list: list[str] = json.loads(Path(asins_path).read_text())
    embeddings: np.ndarray = np.load(emb_path)       # (N, 384) normalised
    asin_to_idx: dict[str, int] = {a: i for i, a in enumerate(asin_list)}
    valid_asins = [a for a in cat.products if a in asin_to_idx]
    valid_idx = [asin_to_idx[a] for a in valid_asins]
    emb_matrix = embeddings[valid_idx]               # (M, 384) — products we have catalog text for
    M = len(valid_asins)
    print(f"  {M:,} products with embeddings")

    # Pre-compute token frequencies across the full catalog (baseline)
    print("Building vocabulary baseline …")
    product_tokens: dict[str, list[str]] = {}
    token_count: Counter[str] = Counter()
    for asin in valid_asins:
        toks = _tokens(_product_text(cat.get(asin, {})))
        product_tokens[asin] = toks
        token_count.update(set(toks))   # count products, not occurrences

    baseline_freq: dict[str, float] = {tok: cnt / M for tok, cnt in token_count.items()}

    print(f"Loading embedding model {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Computing expansions for {len(seed_list)} seeds …")
    result: dict[str, list[str]] = {}

    # Encode all seeds in one batch — efficient
    encoded = model.encode(
        [EMBED_QUERY_PREFIX + s for s in seed_list],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype("float32")

    for seed, qvec in zip(seed_list, encoded):
        # Cosine similarity: qvec (384,) × emb_matrix (M, 384)^T → (M,)
        sims = emb_matrix @ qvec

        top_k_actual = min(top_k, M)
        top_indices = np.argpartition(sims, -top_k_actual)[-top_k_actual:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        # Vocabulary frequency in the top-K neighbours
        neighbor_token_count: Counter[str] = Counter()
        for idx in top_indices:
            asin = valid_asins[idx]
            neighbor_token_count.update(set(product_tokens[asin]))

        # Lift = (freq in neighbours) / (freq in full catalog)
        # Keep terms that are distinctively associated with the seed concept
        seed_tokens = set(_tokens(seed))   # don't expand to the seed's own words
        scored: list[tuple[float, str]] = []
        for tok, cnt in neighbor_token_count.items():
            if tok in seed_tokens:
                continue
            neighbor_freq = cnt / top_k_actual
            if neighbor_freq < min_freq:
                continue
            base = baseline_freq.get(tok, 1 / M)
            lift = neighbor_freq / base
            if lift >= min_lift:
                scored.append((lift, tok))

        scored.sort(reverse=True)
        expansion = [tok for _, tok in scored[:top_n]]
        if expansion:
            result[seed] = expansion

    # Merge with existing synonyms.json if present (don't wipe hand-written entries)
    out = Path(out_path)
    existing: dict[str, list[str]] = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:
            pass
    merged = {**existing, **result}   # data-driven overwrites old data-driven; hand-written in EXPANSIONS is separate

    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {len(merged):,} total entries to {out_path} ({len(result)} new/updated)")

    # Print sample results
    show = ["warm", "waterproof", "vegan", "sustainable", "merino", "oversized",
            "compression", "yoga", "business casual", "anti odor", "high waist"]
    print("\nSample expansions:")
    for s in show:
        if s in result:
            print(f"  {s:22s} → {result[s]}")
        elif s in merged:
            print(f"  {s:22s} → {merged[s]}  (from prior run)")
        else:
            print(f"  {s:22s} → (no result)")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--emb", default="cache/embeddings.npy")
    ap.add_argument("--asins", default="cache/asins.json")
    ap.add_argument("--out", default="cache/synonyms.json")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--min-lift", type=float, default=2.5)
    ap.add_argument("--min-freq", type=float, default=0.12)
    ap.add_argument("--seeds", default="", help="comma-separated seeds (default: full SEEDS list)")
    args = ap.parse_args()

    build(
        catalog_path=args.catalog,
        emb_path=args.emb,
        asins_path=args.asins,
        out_path=args.out,
        top_k=args.top_k,
        top_n=args.top_n,
        min_lift=args.min_lift,
        min_freq=args.min_freq,
        seeds=[s.strip() for s in args.seeds.split(",") if s.strip()] or None,
    )


if __name__ == "__main__":
    main()
