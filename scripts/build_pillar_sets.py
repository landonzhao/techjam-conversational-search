"""Build pillar-targeted, low-leak test sets (see docs/TEST_DESIGN.md).

Reuses the leak-free machinery in build_language_stress_set.py (held-out paraphrase vocab verified
disjoint from catalog text + our synonym tables, plus one fair identifying anchor per session) and
shapes each session so it actually exercises its pillar:

  buying          — the strong reworded constraint is the FIRST hard_constraint (disclosed turn 1).
  browsing        — hard_constraints hold ONLY the anchor; the discriminating attributes sit in
                    soft_preferences, so they are unlocked only when the agent asks the right slot
                    (this is the clarification/MTTC stress).
  intent_override — a soft preference is replaced on turn 3/4 by a new reworded constraint.
  boundary        — scenario_type=boundary; the evaluator's customer declines the first asked
                    attribute, testing graceful "no preference" handling.

--leak controls verbatim leakage:  free (all attrs reworded) | moderate (half reworded).

Output: data/pillar_<leak>.jsonl  + a per-pillar leak report.
Deterministic (seeded), no LLM, no network.

Usage:  python scripts/build_pillar_sets.py [--n 240] [--leak free|moderate] [--seed 11]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index, classify_constraint, coarse_category, intent_card, searchable_text,
)
from evaluator.robustness import _verify_disjoint
from scripts.build_language_stress_set import (
    _ANCHOR_TEMPLATES, _RELATIVE, _TEMPLATES, _content_tokens, discriminator, paraphrasable_attrs,
)

# official scenario mix (40/40/15/5), expanded to whole sessions
SCENARIOS = ["buying"] * 40 + ["browsing"] * 40 + ["intent_override"] * 15 + ["boundary"] * 5


def verbatim_phrase(slot: str, product: dict) -> str | None:
    """A leaky (verbatim) phrase for `slot` lifted from the target — the value the classifier keys
    on, worded as a shopper might. Used for the 'moderate' leak level and as a fallback."""
    blob = searchable_text(product).lower()
    vocab = {
        "material": ("cotton", "polyester", "nylon", "leather", "wool", "silk", "rayon"),
        "color": ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "purple"),
        "use_case": ("hiking", "running", "gym", "winter", "outdoor", "work"),
    }.get(slot)
    if not vocab:
        return None
    for w in vocab:
        if re.search(rf"\b{w}\b", blob):
            return {"material": f"made of {w}", "color": f"in {w}",
                    "use_case": f"for {w}"}[slot]
    return None


def build_attr_phrases(product: dict, attrs: dict[str, str], leak: str, blob_tokens: set[str],
                       rng: random.Random) -> list[str]:
    """Phrases for the target's attributes at the requested leak level. 'free' = held-out
    paraphrase (no catalog tokens); 'moderate' = every other attribute kept verbatim."""
    phrases: list[str] = []
    slots = list(attrs)
    rng.shuffle(slots)
    for i, slot in enumerate(slots):
        vb = verbatim_phrase(slot, product) if leak == "moderate" and i % 2 == 0 else None
        if vb:
            phrases.append(vb)
            continue
        val = attrs[slot]
        if _content_tokens(val) & blob_tokens:      # keep leak-free phrases actually leak-free
            continue
        phrases.append(rng.choice(_TEMPLATES[slot]).format(v=val))
    return phrases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--leak", choices=["free", "moderate"], default="free")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    _verify_disjoint()
    rng = random.Random(args.seed)
    _ids, categories, products = catalog_index(args.catalog)

    by_cat: dict[str, list[str]] = defaultdict(list)
    for asin in products:
        by_cat[coarse_category(categories.get(asin, []))].append(asin)
    cats = list(by_cat)
    rng.shuffle(cats)

    sessions: list[dict] = []
    leak_by_pillar: dict[str, list[float]] = defaultdict(list)
    seen: set[str] = set()
    ci = 0
    while len(sessions) < args.n and ci < len(cats) * 8:
        cat = cats[ci % len(cats)]
        ci += 1
        target = rng.choice(by_cat[cat])
        if target in seen:
            continue
        product = products[target]
        attrs = paraphrasable_attrs(product)
        anchor = discriminator(product)
        if len(attrs) < 2 or not anchor:
            continue
        blob_tokens = _content_tokens(searchable_text(product))
        phrases = build_attr_phrases(product, attrs, args.leak, blob_tokens, rng)
        if len(phrases) < 2:
            continue
        seen.add(target)
        anchor_phrase = rng.choice(_ANCHOR_TEMPLATES).format(d=anchor)
        scenario = SCENARIOS[len(sessions) % len(SCENARIOS)]

        # per-pillar shaping of hard vs soft (what is disclosed vs what must be clarified)
        behavior: dict = {"scenario_type": scenario}
        if scenario == "buying":
            hard, soft = [phrases[0], anchor_phrase], phrases[1:3]
        elif scenario == "browsing":
            # withhold the discriminators: only the anchor is "hard"; attrs unlocked via clarify
            hard, soft = [anchor_phrase], phrases[:3]
        elif scenario == "intent_override":
            hard, soft = [anchor_phrase, phrases[0]], phrases[1:2]
            behavior["override"] = {
                "turn": rng.choice([3, 4]),
                "old_value": soft[0] if soft else "my earlier preference",
                "new_value": phrases[-1],
                "message": f"Actually, ignore that. What I really need is: {phrases[-1]}.",
            }
        else:  # boundary
            hard, soft = [anchor_phrase, phrases[0]], phrases[1:3]

        card = {"target_category": cat, "hard_constraints": hard, "soft_preferences": soft}
        sessions.append({
            "sample_id": f"pillar_{args.leak}_{len(sessions):05d}",
            "scenario_type": scenario,
            "ground_truth": {"parent_asin": target},
            "user_profile": {"preference_tags": [], "summary": ""},
            "intent_card": card, "behavior": behavior,
            "category_bucket": cat, "held_out": args.leak == "free",
        })
        # leak = fraction of attribute-phrase content tokens that appear verbatim in the target
        attr_only = [p for p in (hard + soft) if p != anchor_phrase]
        toks = set().union(*(_content_tokens(p) for p in attr_only)) if attr_only else set()
        leak_by_pillar[scenario].append(len(toks & blob_tokens) / max(1, len(toks)))

    out = Path(f"data/pillar_{args.leak}.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for s in sessions:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    counts = defaultdict(int)
    for s in sessions:
        counts[s["scenario_type"]] += 1
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    print(f"wrote {len(sessions)} sessions -> {out}  (leak={args.leak})")
    print(f"distinct categories: {len({s['category_bucket'] for s in sessions})}")
    print("per-pillar count | attribute leak rate:")
    for pil in ("buying", "browsing", "intent_override", "boundary"):
        print(f"  {pil:>16}: n={counts[pil]:>3}  leak={mean(leak_by_pillar[pil]):.3f}")
    # sanity: how the browsing discriminators classify (which slot the agent must ask)
    demo = next(s for s in sessions if s["scenario_type"] == "browsing")
    print("\nsample browsing card:", json.dumps(demo["intent_card"], ensure_ascii=False))
    print("  its soft prefs classify as:",
          [classify_constraint(p) for p in demo["intent_card"]["soft_preferences"]])


if __name__ == "__main__":
    main()
