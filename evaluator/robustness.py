"""P8 — de-rigged robustness harness (held-out paraphrase evaluation).

WHY THIS EXISTS
---------------
The public evaluator leaks: `intent_card()` builds the disclosed constraints VERBATIM from the
target product's own `features`/`details`, so our `CoverageReranker` can win by literal token
coverage. That inflates the offline score but says nothing about a REAL shopper who paraphrases
("keeps the rain out" instead of the catalog's "waterproof"). Every "neutral" measurement we
have is on that leaked distribution.

METHOD (two modes)
------------------
* mode="paraphrase" (DEFAULT, the fair test): run the OFFICIAL disclosure loop unchanged —
  identical information, identical override/boundary cadence — but pass every shopper message
  through `paraphrase()`, which rewords known attribute tokens (material/color/use-case/feature)
  into a **held-out** vocabulary. Discriminating tokens (brand, distinctive nouns, numbers)
  survive; only the words coverage keys on change. This isolates the *leak* from information.
* mode="reduced": an aggressive floor — describe the target with only a few vague paraphrased
  attributes (also strips discriminating info). Useful as a lower bound, not a fair score.

The paraphrase vocabulary is asserted **disjoint from `src.understanding.EXPANSIONS`** at startup,
so we are not grading our own synonym table against itself (that would be circular).

Metrics mirror the official evaluator (hit@10 / MRR / MTTC, break at first top-10 appearance).

Usage:
    python -m evaluator.robustness                 # fair paraphrase mode, all ablations
    python -m evaluator.robustness --mode reduced
    python -m evaluator.robustness --limit 60
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply, initial_message,
    load_jsonl, materialize_hidden_fields, normalize_recommendations,
)
from src.understanding import COLOR_RE, EXPANSIONS, MATERIAL_RE, USE_CASE_KEYS, resolve_category

# --------------------------------------------------------------------------- held-out lexicon
# value/trigger -> a natural phrase a real shopper might say. Replacement phrases avoid every
# key/value token in EXPANSIONS (verified at runtime) so the agent can win by neither verbatim
# coverage NOR our own seed synonyms. Triggers MAY be EXPANSIONS keys — we are removing them.
MATERIAL_PARAPHRASE = {
    "cotton": "a natural plain cloth", "polyester": "a synthetic man-made fabric",
    "leather": "genuine animal hide", "nylon": "a thin plasticky weave",
    "wool": "a thick fuzzy knit for the cold", "denim": "a firm jean-style cloth",
    "silk": "a smooth glossy material", "linen": "a crisp summery weave",
    "suede": "a napped matte hide", "spandex": "a clingy form-fitting material",
    "cashmere": "a plush luxury knit", "velvet": "a rich napped pile",
    "fleece": "a fuzzy pile knit", "rayon": "a silky synthetic drape",
}
USECASE_PARAPHRASE = {
    "hiking": "long treks over rocky ground", "running": "fast-paced pavement exercise",
    "gym": "indoor strength sessions", "workout": "sweaty exercise routines",
    "winter": "freezing snowy weather", "summer": "hot sunny afternoons",
    "beach": "sandy seaside trips", "formal": "black-tie occasions",
    "office": "the professional workplace", "wedding": "a marriage ceremony",
    "travel": "long trips away from home", "outdoor": "spending the day outside",
    "rain": "wet drizzly days", "party": "a lively evening event",
}
COLOR_PARAPHRASE = {
    "black": "a very dark shade", "white": "a pale bright shade", "blue": "an ocean-like hue",
    "red": "a fiery bold hue", "green": "a leafy hue", "gray": "a neutral ashen shade",
    "grey": "a neutral ashen shade", "navy": "a deep midnight shade", "brown": "an earthy tone",
    "pink": "a rosy tone", "purple": "a violet tone", "orange": "a citrus tone",
}
# common discriminating feature adjectives coverage keys on (triggers may be EXPANSIONS keys)
FEATURE_PARAPHRASE = {
    "waterproof": "able to keep rain out", "water-resistant": "good at fending off wet weather",
    "insulated": "built to trap body heat", "breathable": "able to let air pass through",
    "lightweight": "very easy to carry", "durable": "long-lasting and tough",
    "adjustable": "able to be resized", "wireless": "with no cords",
    "moisture-wicking": "quick to pull sweat away", "waterproofing": "a barrier against rain",
    "stretchy": "able to give and flex", "cushioned": "padded underfoot",
    "quick-dry": "fast to dry off", "windproof": "able to block gusts",
}

_ALL = {**MATERIAL_PARAPHRASE, **USECASE_PARAPHRASE, **COLOR_PARAPHRASE, **FEATURE_PARAPHRASE}
# longest triggers first so multi-word ones win (e.g. "water-resistant" before "water")
_SUBS = sorted(_ALL.items(), key=lambda kv: -len(kv[0]))
_LOOKUP = {t.lower(): p for t, p in _SUBS}
_PATTERN = re.compile(r"\b(" + "|".join(re.escape(t) for t, _ in _SUBS) + r")\b", re.IGNORECASE)


def _verify_disjoint() -> None:
    exp_tokens: set[str] = set()
    for k, vs in EXPANSIONS.items():
        exp_tokens.update(re.findall(r"[a-z]+", k))
        for v in vs:
            exp_tokens.update(re.findall(r"[a-z]+", v))
    overlap = {tok for phrase in _ALL.values() for tok in re.findall(r"[a-z]+", phrase)
               if tok in exp_tokens}
    if overlap:
        raise SystemExit(f"CIRCULAR TEST: paraphrase tokens overlap EXPANSIONS: {sorted(overlap)}")
    print(f"held-out vocab verified disjoint from EXPANSIONS "
          f"({len(exp_tokens)} expansion tokens, {len(_ALL)} paraphrases, 0 overlap)\n")


def paraphrase(text: str) -> str:
    """Reword known attribute tokens into held-out language in a SINGLE pass (no cascade, so an
    inserted replacement is never re-substituted); leave everything else intact."""
    return _PATTERN.sub(lambda m: _LOOKUP[m.group(0).lower()], text)


# --------------------------------------------------------------------------- reduced mode
def _first(rx: re.Pattern, text: str) -> str | None:
    m = rx.search(text)
    return m.group(1).lower() if m else None


def reduced_turns(product: dict) -> list[str]:
    title = str(product.get("title") or "")
    blob = f"{title} {' '.join(map(str, product.get('features') or []))} {product.get('details') or ''}".lower()
    cat = resolve_category(title.lower()) or "item"
    uc = next((k for k in USE_CASE_KEYS if re.search(rf"\b{k}\b", blob)), None)
    mat = _first(MATERIAL_RE, blob)
    col = _first(COLOR_RE, title.lower())
    turns = [f"I need a {cat} for {USECASE_PARAPHRASE[uc]}." if uc in USECASE_PARAPHRASE
             else f"I'm looking for a {cat}."]
    if mat in MATERIAL_PARAPHRASE:
        turns.append(f"Ideally it's made of {MATERIAL_PARAPHRASE[mat]}.")
    if col in COLOR_PARAPHRASE:
        turns.append(f"I'd prefer {COLOR_PARAPHRASE[col]}.")
    return turns


# --------------------------------------------------------------------------- evaluation loop
def run(agent, samples, catalog_ids, categories, products, mode: str) -> dict:
    rrs, hits, first_turns, usable = [], [], [], 0
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        sid = f"robust_{sample['sample_id']}"
        agent.reset(sid, sample.get("user_profile", {}))
        best_rank = first_turn = None

        if mode == "reduced":
            turns = reduced_turns(products.get(target, {}))
            if len(turns) < 1:
                continue
            usable += 1
            for turn in range(1, MAX_TURNS + 1):
                msg = turns[turn - 1] if turn <= len(turns) else turns[-1]
                try:
                    ranked = normalize_recommendations(
                        agent.respond(sid, msg, turn, TOP_K).get("recommendations"), catalog_ids)
                except Exception:
                    ranked = []
                if target in ranked:
                    best_rank, first_turn = ranked.index(target) + 1, turn
                    break
        else:  # paraphrase: official flow, reworded messages (same information)
            usable += 1
            eic, eb = materialize_hidden_fields(sample, products)
            es = {**sample, "intent_card": eic, "behavior": eb}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = sample["scenario_type"] != "intent_override"
            msg = initial_message(es, coarse_category(categories.get(target, [])), disclosed)
            for turn in range(1, MAX_TURNS + 1):
                try:
                    ranked = normalize_recommendations(
                        agent.respond(sid, paraphrase(msg), turn, TOP_K).get("recommendations"),
                        catalog_ids)
                except Exception:
                    ranked = []
                if override_applied and target in ranked:
                    best_rank, first_turn = ranked.index(target) + 1, turn
                    break
                if turn == MAX_TURNS:
                    break
                ov = es.get("behavior", {}).get("override") or {}
                if not override_applied and turn + 1 == int(ov.get("turn", 3)):
                    override_applied = True
                    nv = str(ov.get("new_value", ""))
                    if nv:
                        disclosed.add(nv)
                    msg = str(ov.get("message", "Actually, ignore my earlier preference."))
                else:
                    msg, boundary_used = customer_reply(
                        es, "other", disclosed, boundary_used)  # drain constraints (matches full run)

        hits.append(1 if best_rank else 0)
        rrs.append(0.0 if not best_rank else 1.0 / best_rank)
        first_turns.append(first_turn if first_turn else MAX_TURNS + 1)

    hit = statistics.fmean(hits) if hits else 0.0
    mrr = statistics.fmean(rrs) if rrs else 0.0
    mttc = statistics.fmean(first_turns) if first_turns else 0.0
    eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {"n": usable, "hit@10": hit, "mrr": mrr, "mttc": mttc,
            "score": 0.50 * hit + 0.30 * mrr + 0.20 * eff}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--dataset", default="data/public_set.jsonl")
    ap.add_argument("--mode", choices=["paraphrase", "reduced"], default="paraphrase")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    _verify_disjoint()
    from src.agent import Agent

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = catalog_index(args.catalog)

    configs = {
        "full (understanding+coverage)":        dict(),
        "no coverage (leak removed)":           dict(USE_COVERAGE_RERANK=False),
        "no understanding (expand+usecase off)": dict(USE_SLOT_EXPANSION=False, USE_USECASE_PRIORS=False),
        "backbone only (no cov, no underst.)":  dict(USE_COVERAGE_RERANK=False, USE_SLOT_EXPANSION=False,
                                                     USE_USECASE_PRIORS=False),
    }
    print(f"mode={args.mode}   (official offline score with the leak = 0.8840)\n")
    print(f"{'config':38} {'n':>4} {'hit@10':>7} {'mrr':>7} {'mttc':>6} {'score':>7}")
    print("-" * 74)
    for name, overrides in configs.items():
        saved = {k: getattr(Agent, k) for k in overrides}
        for k, v in overrides.items():
            setattr(Agent, k, v)
        try:
            res = run(Agent(args.catalog), samples, catalog_ids, categories, products, args.mode)
        finally:
            for k, v in saved.items():
                setattr(Agent, k, v)
        print(f"{name:38} {res['n']:>4} {res['hit@10']:>7.3f} {res['mrr']:>7.3f} "
              f"{res['mttc']:>6.2f} {res['score']:>7.4f}")


if __name__ == "__main__":
    main()
