"""Generate synthetic evaluation sets that mimic data/public_set.jsonl.

The official evaluator (evaluator/local_evaluator.py) derives the whole
simulated conversation -- intent card, hard/soft constraints and override
behaviour -- directly from the ground-truth product. A sample therefore only
needs three real fields to be fully evaluable:

    * ground_truth.parent_asin  (must exist in the frozen catalog)
    * scenario_type             (buying | browsing | intent_override | boundary)
    * user_profile              (same shape as the public set)

Everything else (category_bucket, difficulty_bucket) is a stratification label.

This script produces additional held-out sets over the SAME frozen 50k catalog
with far more diverse user profiles, target items and scenario mixes than the
200-sample public set, without overfitting to public session ids: it uses a
`synth_` id prefix and excludes every public ground-truth ASIN by default.

Usage:
    python scripts/generate_synthetic_set.py                       # 1000 samples
    python scripts/generate_synthetic_set.py --count 500 --seed 7
    python scripts/generate_synthetic_set.py --output data/synthetic_hard.jsonl

Validate a generated set against the evaluator with:
    python scripts/measure.py --dataset data/synthetic_set.jsonl --limit 40
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

# Scenario mix mirrors the public set proportions by default
# (buying 40%, browsing 40%, intent_override 15%, boundary 5%).
DEFAULT_SCENARIO_WEIGHTS = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}

# Vocabulary is a superset of what appears in the public set so profiles stay
# in-distribution while covering more ground.
PREFERENCE_TAGS = [
    "fit", "comfort", "durability", "style", "material", "weather", "warmth",
    "performance", "breathability", "value", "brand", "versatility",
    "sustainability", "color", "sizing",
]

PURCHASE_FREQUENCIES = [
    "first-time shopper",
    "1-2 prior purchases",
    "3-4 prior purchases",
    "5-8 prior purchases",
    "frequent buyer (9+ prior purchases)",
]

# rating_style is kept coherent with average_prior_rating.
RATING_PROFILES = [
    ("critical", [1.0, 2.0]),
    ("mixed", [2.0, 3.0, 4.0]),
    ("usually positive", [4.0, 5.0]),
    ("enthusiast", [5.0]),
]

EXCLUDED_LEVEL2 = {"clothing, shoes & jewelry", "clothing shoes & jewelry", "clothing"}


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gender_group(product: dict) -> str:
    for value in product.get("categories") or []:
        low = str(value).strip().lower()
        if low in EXCLUDED_LEVEL2:
            continue
        if low in {"women", "men", "girls", "boys", "baby", "unisex"}:
            return low
    return "general"


def category_bucket(product: dict) -> str:
    """A readable label from the deepest meaningful category level."""
    cleaned = [
        str(v).strip()
        for v in product.get("categories") or []
        if str(v).strip().lower() not in EXCLUDED_LEVEL2 and str(v).strip()
    ]
    return " > ".join(cleaned[-2:]).lower() if cleaned else "clothing"


def price_band(product: dict) -> str:
    price = product.get("price")
    try:
        value = float(price)
    except (TypeError, ValueError):
        return "unpriced"
    if value < 20:
        return "budget"
    if value < 60:
        return "mid"
    return "premium"


def popularity_band(product: dict, easy_cut: float, hard_cut: float) -> str:
    count = product.get("rating_number") or 0
    if count >= easy_cut:
        return "easy"
    if count <= hard_cut:
        return "hard"
    return "medium"


def build_profile(rng: random.Random, product: dict) -> dict:
    style, rating_pool = rng.choice(RATING_PROFILES)
    average = rng.choice(rating_pool)
    tags = rng.sample(PREFERENCE_TAGS, rng.randint(2, 4))
    frequency = rng.choice(PURCHASE_FREQUENCIES)
    summary = (
        f"Prior purchases emphasize {', '.join(tags)}; "
        f"ratings are {style if style != 'enthusiast' else 'consistently glowing'}."
    )
    return {
        "average_prior_rating": average,
        "preference_tags": tags,
        "purchase_frequency": frequency,
        "rating_style": style,
        "summary": summary,
    }


def stratified_products(
    products: list[dict], count: int, rng: random.Random
) -> list[dict]:
    """Round-robin over (gender, price, popularity) buckets for even coverage."""
    counts = sorted(p.get("rating_number") or 0 for p in products)
    easy_cut = counts[int(len(counts) * 0.75)] if counts else 0
    hard_cut = counts[int(len(counts) * 0.25)] if counts else 0

    buckets: dict[tuple[str, str, str], list[dict]] = {}
    for product in products:
        key = (
            gender_group(product),
            price_band(product),
            popularity_band(product, easy_cut, hard_cut),
        )
        buckets.setdefault(key, []).append(product)
    for pool in buckets.values():
        rng.shuffle(pool)

    order = list(buckets.keys())
    rng.shuffle(order)
    selected: list[dict] = []
    cursor = 0
    while len(selected) < count and any(buckets.values()):
        pool = buckets[order[cursor % len(order)]]
        if pool:
            selected.append(pool.pop())
        cursor += 1
    return selected[:count]


def choose_scenarios(count: int, weights: dict[str, float], rng: random.Random) -> list[str]:
    names = list(weights)
    probs = [weights[name] for name in names]
    scenarios = rng.choices(names, weights=probs, k=count)
    return scenarios


def generate(
    catalog_path: str,
    public_path: str,
    count: int,
    seed: int,
    id_prefix: str,
    weights: dict[str, float],
    gender: str | None = None,
    price: str | None = None,
    difficulty: str | None = None,
) -> list[dict]:
    """Build a synthetic set. Optional filters carve focused slices:

    gender     : women | men | girls | boys | baby | unisex | general
    price      : budget | mid | premium | unpriced
    difficulty : easy | medium | hard  (popularity band, global cuts)
    """
    rng = random.Random(seed)
    catalog = load_jsonl(catalog_path)

    excluded = set()
    if Path(public_path).exists():
        excluded = {
            str(s["ground_truth"]["parent_asin"])
            for s in load_jsonl(public_path)
            if s.get("ground_truth")
        }
    candidates = [p for p in catalog if str(p["parent_asin"]) not in excluded]

    # Global popularity cuts define difficulty labels consistently across slices.
    counts = sorted(p.get("rating_number") or 0 for p in candidates)
    easy_cut = counts[int(len(counts) * 0.75)] if counts else 0
    hard_cut = counts[int(len(counts) * 0.25)] if counts else 0

    if gender is not None:
        candidates = [p for p in candidates if gender_group(p) == gender]
    if price is not None:
        candidates = [p for p in candidates if price_band(p) == price]
    if difficulty is not None:
        candidates = [p for p in candidates
                      if popularity_band(p, easy_cut, hard_cut) == difficulty]
    if not candidates:
        raise SystemExit("no catalog products match the requested filters")

    chosen = stratified_products(candidates, count, rng)
    scenarios = choose_scenarios(len(chosen), weights, rng)

    samples: list[dict] = []
    for index, (product, scenario) in enumerate(zip(chosen, scenarios), start=1):
        samples.append({
            "category_bucket": category_bucket(product),
            "difficulty_bucket": popularity_band(product, easy_cut, hard_cut),
            "ground_truth": {"parent_asin": str(product["parent_asin"])},
            "sample_id": f"{id_prefix}{index:05d}",
            "scenario_type": scenario,
            "user_profile": build_profile(rng, product),
        })
    return samples


def summarize(samples: list[dict]) -> str:
    from collections import Counter

    scen = Counter(s["scenario_type"] for s in samples)
    diff = Counter(s["difficulty_bucket"] for s in samples)
    gender = Counter(s["category_bucket"].split(" > ")[0] for s in samples)
    unique = len({s["ground_truth"]["parent_asin"] for s in samples})
    ratings = [s["user_profile"]["average_prior_rating"] for s in samples]
    lines = [
        f"samples: {len(samples)}  unique targets: {unique}",
        f"scenario: {dict(scen)}",
        f"difficulty: {dict(diff)}",
        f"top category leads: {dict(gender.most_common(6))}",
        f"avg_prior_rating mean: {statistics.fmean(ratings):.2f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl",
                        help="public set whose ground-truth ASINs are excluded (held-out)")
    parser.add_argument("--output", default="data/synthetic_set.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-prefix", default="synth_")
    parser.add_argument("--gender", choices=["women", "men", "girls", "boys", "baby", "unisex", "general"])
    parser.add_argument("--price", choices=["budget", "mid", "premium", "unpriced"])
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--scenario", choices=list(DEFAULT_SCENARIO_WEIGHTS),
                        help="restrict to a single scenario type")
    parser.add_argument("--include-public-targets", action="store_true",
                        help="do NOT exclude public ground-truth ASINs")
    args = parser.parse_args()

    weights = {args.scenario: 1.0} if args.scenario else DEFAULT_SCENARIO_WEIGHTS
    public_path = "" if args.include_public_targets else args.public
    samples = generate(
        catalog_path=args.catalog,
        public_path=public_path or "___none___",
        count=args.count,
        seed=args.seed,
        id_prefix=args.id_prefix,
        weights=weights,
        gender=args.gender,
        price=args.price,
        difficulty=args.difficulty,
    )

    out = Path(args.output)
    with out.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")
    print(f"wrote {len(samples)} samples -> {out}")
    print(summarize(samples))


if __name__ == "__main__":
    main()
