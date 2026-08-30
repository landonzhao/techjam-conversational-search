"""Build a leak-free, natural-language, long-tail-stressing held-out session set.

WHY
---
The public set (and our synthetic set) leak: the evaluator materializes each brief VERBATIM from
the target's own spec sheet, so verbatim-coverage wins without understanding anything. Every
"neutral" measurement we have is on that leaked distribution. This generator produces sessions that
carry their OWN `intent_card` — so the evaluator uses it directly (see materialize_hidden_fields) —
with constraints written in HELD-OUT natural phrasing that does not appear in the target's catalog
text. Running the standard evaluator on this set therefore measures generalization, not leak
exploitation, and predicts the private set far better than the public score.

THREE STRESSES
--------------
1. Leak-free   — attribute values are reworded into a vocabulary verified disjoint from both the
   catalog text (per-target check) and our own EXPANSIONS (via evaluator.robustness).
2. Natural     — vague / relative / multi-attribute phrasings across many templates, not spec strings.
3. Long tail   — targets are sampled across DISTINCT coarse categories, over-weighting the obscure
   buckets where our category-blind questioning degrades.

Output: data/language_stress_set.jsonl  (+ a summary with the measured leak rate vs the leaky brief).
Deterministic (seeded), no LLM, no network — reproducible.

Usage:  python scripts/build_language_stress_set.py [--n 250] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import catalog_index, coarse_category, intent_card, searchable_text
from evaluator.robustness import (
    COLOR_PARAPHRASE, FEATURE_PARAPHRASE, MATERIAL_PARAPHRASE, USECASE_PARAPHRASE, _verify_disjoint,
)
from src.understanding import COLOR_RE, MATERIAL_RE, USE_CASE_KEYS

_WORD = re.compile(r"[a-z0-9]+")
# generic template words that are not discriminating signal (excluded from the leak check)
_STOP = {
    "something", "made", "of", "for", "the", "a", "an", "it", "in", "with", "that", "is", "to",
    "and", "but", "not", "too", "ideally", "prefer", "id", "want", "need", "should", "feel",
    "like", "im", "looking", "mainly", "use", "meant", "keep", "near", "roughly", "my", "be",
    "wear", "wearing", "day", "long", "bit", "more", "than", "everyday", "simple", "nothing",
}

# Natural templates per attribute (multiple, for phrasing diversity). {v} = held-out value phrase.
_TEMPLATES = {
    "material": ["made of {v}", "the material should feel like {v}", "ideally in {v}",
                 "I want it to be {v}"],
    "feature": ["it needs to be {v}", "ideally {v}", "something {v}", "I'd want it {v}"],
    "use_case": ["for {v}", "I'll mainly use it for {v}", "meant for {v}", "to wear during {v}"],
    "color": ["in {v}", "I'd prefer {v}", "color-wise, {v}", "something in {v}"],
}
# Vague / relative soft preferences — held-out language with no catalog signal, adds dialogue noise.
_RELATIVE = [
    "nothing too flashy", "keep it fairly understated", "a bit dressier than everyday",
    "not too bulky", "something that feels premium", "easy to pair with other things",
    "comfortable for a long day", "on the simpler side",
]
# Natural ways a real shopper names a distinctive anchor (brand / distinctive noun). {d} = anchor.
_ANCHOR_TEMPLATES = [
    "I think it was a {d}", "something from {d}", "a {d} one", "the {d} kind", "by {d}"]

# Words that are NOT distinctive anchors: generic attributes/categories/stopwords.
_GENERIC = _STOP | set(MATERIAL_PARAPHRASE) | set(COLOR_PARAPHRASE) | set(USECASE_PARAPHRASE) | {
    "womens", "mens", "kids", "girls", "boys", "unisex", "size", "small", "medium", "large",
    "set", "pack", "pair", "new", "style", "fashion", "classic", "premium", "quality",
}


def discriminator(product: dict) -> str | None:
    """A legitimate identifying anchor a real shopper would give (brand or a distinctive title
    noun). This is FAIR signal — it may appear in catalog text — as opposed to the attribute words
    coverage exploits, which we de-leak. Without an anchor the target is unidentifiable among many
    look-alikes (that is 'reduced' mode, an unfair floor)."""
    store = str(product.get("store") or "").strip()
    if 2 < len(store) <= 30 and _WORD.search(store):
        return store
    title = str(product.get("title") or "")
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", title):
        if tok.lower() not in _GENERIC:
            return tok
    return None


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 2 and t not in _STOP}


def _first(rx: re.Pattern, text: str) -> str | None:
    m = rx.search(text)
    return m.group(1).lower() if m else None


def paraphrasable_attrs(product: dict) -> dict[str, str]:
    """Return {slot: held-out phrase} for every attribute of `product` we can reword out of the
    catalog's vocabulary. Only attributes with a held-out paraphrase are eligible, so every
    generated constraint carries real, non-leaked signal."""
    blob = searchable_text(product).lower()
    title = str(product.get("title") or "").lower()
    out: dict[str, str] = {}
    mat = _first(MATERIAL_RE, blob)
    if mat in MATERIAL_PARAPHRASE:
        out["material"] = MATERIAL_PARAPHRASE[mat]
    col = _first(COLOR_RE, title)
    if col in COLOR_PARAPHRASE:
        out["color"] = COLOR_PARAPHRASE[col]
    uc = next((k for k in USE_CASE_KEYS if re.search(rf"\b{k}\b", blob)), None)
    if uc in USECASE_PARAPHRASE:
        out["use_case"] = USECASE_PARAPHRASE[uc]
    feat = next((k for k in FEATURE_PARAPHRASE if re.search(rf"\b{re.escape(k)}\b", blob)), None)
    if feat:
        out["feature"] = FEATURE_PARAPHRASE[feat]
    return out


def build_constraints(attrs: dict[str, str], anchor: str, blob_tokens: set[str],
                      rng: random.Random) -> tuple[list[str], list[str], list[str]]:
    """Build natural hard/soft constraints: a discriminating ANCHOR (fair identifying signal) plus
    held-out attribute paraphrases (de-leaked). Returns (hard, soft, attr_phrases) — attr_phrases is
    the de-leaked subset, tracked so we can report attribute leak separately from the anchor."""
    attr_phrases: list[str] = []
    slots = list(attrs)
    rng.shuffle(slots)
    for slot in slots:
        value = attrs[slot]
        # anti-leak: the held-out attribute phrase's content tokens must not appear in target text
        if _content_tokens(value) & blob_tokens:
            continue
        attr_phrases.append(rng.choice(_TEMPLATES[slot]).format(v=value))
    if len(attr_phrases) >= 3 and rng.random() < 0.5:
        # occasionally fuse two attributes into one multi-attribute message (dialogue stress)
        attr_phrases[0] = f"{attr_phrases[0]}, and {attr_phrases[1]}"
        attr_phrases.pop(1)
    anchor_phrase = rng.choice(_ANCHOR_TEMPLATES).format(d=anchor)
    # hard: anchor + strongest attribute; soft: remaining attributes + a relative preference
    hard = [anchor_phrase] + attr_phrases[:1]
    soft = attr_phrases[1:3]
    rel = rng.choice(_RELATIVE)
    if not (_content_tokens(rel) & blob_tokens):
        soft.append(rel)
    return hard, soft[:2], attr_phrases


def leak_rate(constraints: list[str], blob_tokens: set[str]) -> float:
    toks = set().union(*(_content_tokens(c) for c in constraints)) if constraints else set()
    return len(toks & blob_tokens) / max(1, len(toks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--out", default="data/language_stress_set.jsonl")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    _verify_disjoint()  # held-out vocab is disjoint from our EXPANSIONS (no circular grading)
    rng = random.Random(args.seed)
    _ids, categories, products = catalog_index(args.catalog)

    # Group products by coarse category, then sample ACROSS distinct categories (long-tail weighting:
    # each obscure bucket contributes, instead of head categories dominating a per-product sample).
    by_cat: dict[str, list[str]] = defaultdict(list)
    for asin in products:
        by_cat[coarse_category(categories.get(asin, []))].append(asin)
    cats = list(by_cat)
    rng.shuffle(cats)

    scenarios = (["buying"] * 40 + ["browsing"] * 35 + ["intent_override"] * 15 + ["boundary"] * 10)
    sessions: list[dict] = []
    leak_ours: list[float] = []
    leak_leaky: list[float] = []
    seen: set[str] = set()

    # round-robin over categories so the tail is over-represented relative to product frequency
    ci = 0
    while len(sessions) < args.n and ci < len(cats) * 6:
        cat = cats[ci % len(cats)]
        ci += 1
        pool = by_cat[cat]
        target = rng.choice(pool)
        if target in seen:
            continue
        product = products[target]
        attrs = paraphrasable_attrs(product)
        anchor = discriminator(product)
        if len(attrs) < 2 or not anchor:   # need held-out signal AND an identifying anchor
            continue
        blob_tokens = _content_tokens(searchable_text(product))
        hard, soft, attr_phrases = build_constraints(attrs, anchor, blob_tokens, rng)
        if len(hard) < 2 or len(attr_phrases) < 1:
            continue
        seen.add(target)

        scenario = scenarios[len(sessions) % len(scenarios)]
        behavior: dict = {"scenario_type": scenario}
        if scenario == "intent_override":
            behavior["override"] = {
                "turn": rng.choice([3, 4]),
                "old_value": soft[-1] if soft else "a different style",
                "new_value": hard[0],
                "message": f"Actually, ignore my earlier preference. What I need is: {hard[0]}.",
            }
        card = {"target_category": coarse_category(categories.get(target, [])),
                "hard_constraints": hard, "soft_preferences": soft}
        sessions.append({
            "sample_id": f"lstress_{len(sessions):05d}",
            "scenario_type": scenario,
            "ground_truth": {"parent_asin": target},
            "user_profile": {"preference_tags": [], "summary": ""},
            "intent_card": card,
            "behavior": behavior,
            "category_bucket": cat,
            "held_out": True,
        })
        # measure leak on the ATTRIBUTE phrases only (the de-leaked signal); the anchor is a
        # deliberate identifying token that legitimately appears in catalog text.
        leak_ours.append(leak_rate(attr_phrases, blob_tokens))
        leaky = intent_card(product)
        leak_leaky.append(leak_rate(
            leaky["hard_constraints"] + leaky["soft_preferences"], blob_tokens))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for s in sessions:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    scen = defaultdict(int)
    for s in sessions:
        scen[s["scenario_type"]] += 1
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    print(f"wrote {len(sessions)} sessions -> {out}")
    print(f"distinct categories covered: {len({s['category_bucket'] for s in sessions})}")
    print(f"scenario mix: {dict(scen)}")
    print(f"attribute leak rate — OURS (de-leaked): {mean(leak_ours):.3f}   "
          f"vs LEAKY brief on same targets: {mean(leak_leaky):.3f}   "
          f"(each session keeps ONE identifying anchor so the target stays findable)")
    print("\nsample session:")
    print(json.dumps(sessions[0]["intent_card"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
