"""Build a wide, labeled suite of held-out evaluation sets over the frozen 50k catalog.

Emits multiple stratified .jsonl sets into data/test_suite/ plus a manifest.json
describing each one (filters, seed, count, path). Every set:
    * uses the same frozen catalog,
    * excludes all public ground-truth ASINs (genuine held-out),
    * uses `synth_`-prefixed ids and an independent seed (reproducible).

The suite spans difficulty, product/gender category, price band and scenario so
you can measure and debug the agent across the full distribution -- not just the
clothing-heavy, findable public 200.

Usage:
    python scripts/build_test_suite.py                    # full suite
    python scripts/build_test_suite.py --outdir data/test_suite --scale 1.0

Evaluate + trace any set with:
    python scripts/trace_eval.py --dataset data/test_suite/hard.jsonl --trace-dir traces/hard
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_synthetic_set import DEFAULT_SCENARIO_WEIGHTS, generate, summarize

# name -> (count, filters). filters keys: gender, price, difficulty, scenario.
SUITE: dict[str, tuple[int, dict]] = {
    # Broad reference set across everything.
    "diverse":        (1000, {}),
    # Difficulty slices (popularity bands) -- isolate the hard tail.
    "easy":           (300, {"difficulty": "easy"}),
    "medium":         (300, {"difficulty": "medium"}),
    "hard":           (300, {"difficulty": "hard"}),
    # Category / audience slices.
    "women":          (300, {"gender": "women"}),
    "men":            (300, {"gender": "men"}),
    "girls":          (200, {"gender": "girls"}),
    "boys":           (200, {"gender": "boys"}),
    # Price-band slices.
    "budget":         (300, {"price": "budget"}),
    "premium":        (300, {"price": "premium"}),
    # Scenario-focused slices (single scenario each).
    "buying_only":    (300, {"scenario": "buying"}),
    "browsing_only":  (300, {"scenario": "browsing"}),
    "override_only":  (200, {"scenario": "intent_override"}),
    "boundary_only":  (200, {"scenario": "boundary"}),
}


def build(outdir: Path, catalog: str, public: str, base_seed: int, scale: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for offset, (name, (count, filters)) in enumerate(SUITE.items()):
        scenario = filters.get("scenario")
        weights = {scenario: 1.0} if scenario else DEFAULT_SCENARIO_WEIGHTS
        n = max(1, int(count * scale))
        seed = base_seed + offset * 1000

        try:
            samples = generate(
                catalog_path=catalog,
                public_path=public,
                count=n,
                seed=seed,
                id_prefix=f"synth_{name}_",
                weights=weights,
                gender=filters.get("gender"),
                price=filters.get("price"),
                difficulty=filters.get("difficulty"),
            )
        except SystemExit as exc:
            print(f"skip {name}: {exc}")
            continue

        path = outdir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample) + "\n")

        manifest.append({
            "name": name,
            "path": str(path),
            "count": len(samples),
            "seed": seed,
            "filters": filters,
        })
        print(f"\n[{name}] -> {path}")
        print("  " + summarize(samples).replace("\n", "\n  "))

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(item["count"] for item in manifest)
    print(f"\nwrote {len(manifest)} sets ({total} samples) + {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl")
    parser.add_argument("--outdir", default="data/test_suite")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="scale every set's count (e.g. 0.2 for a quick suite)")
    args = parser.parse_args()
    build(Path(args.outdir), args.catalog, args.public, args.seed, args.scale)


if __name__ == "__main__":
    main()
