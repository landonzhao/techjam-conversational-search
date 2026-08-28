"""One-time embedding precompute for the Phase 3 vector track.

Embeds each catalog product's ``title + categories + features`` (the 97-100%
filled fields) with BAAI/bge-small-en-v1.5 and writes:
    cache/embeddings.npy   float32 (N, 384), L2-normalized
    cache/asins.json       list[str] aligned row-for-row with the matrix

Running this also populates the local HuggingFace model cache, so the query
encoder works offline afterwards (final scoring may disable network).

Usage:
    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --limit 5000   # quick pipeline test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--out-dir", default="cache")
    parser.add_argument("--limit", type=int, default=0, help="embed first N rows only")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-seq-len", type=int, default=128,
        help="cap token length; title+categories+lead features fit, keeps encoding fast",
    )
    args = parser.parse_args()

    asins: list[str] = []
    texts: list[str] = []
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            p = json.loads(line)
            asins.append(str(p["parent_asin"]))
            texts.append(
                f"{_text(p.get('title'))} "
                f"{_text(p.get('categories'))} "
                f"{_text(p.get('features'))}".strip()
            )
            if args.limit and len(asins) >= args.limit:
                break

    print(f"Loaded {len(asins)} products. Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = args.max_seq_len   # cap sequence cost (features can be long)

    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    print(f"Encoded {len(texts)} in {time.time() - start:.1f}s  shape={embeddings.shape}")

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    np.save(out / "embeddings.npy", embeddings)
    (out / "asins.json").write_text(json.dumps(asins), encoding="utf-8")
    print(f"Wrote {out/'embeddings.npy'} and {out/'asins.json'}")


if __name__ == "__main__":
    main()
