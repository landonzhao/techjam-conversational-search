"""Train the LTR ranker (docs/ADVANCED_RANKING_PLAN.md, step 3).

Reads cache/ltr_data.jsonl (leak-balanced), standardizes features, fits a class-weighted logistic
regression (few params -> generalizes across the distinct private users), reports per-feature weights
and held-out ranking quality (MRR / hit@10 grouped by session), and saves cache/ltr_model.json.

A linear model is deliberate: with hundreds of sessions and a distinct private test population, a
low-variance model that we can read the weights of beats a boosted tree we cannot trust.

Usage:  python -u scripts/train_ltr.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.ranking_features import FEATURE_NAMES

DATA = "cache/ltr_data.jsonl"
MODEL_OUT = "cache/ltr_model.json"


def session_metrics(rows, score_fn) -> tuple[float, float]:
    """MRR and hit@10 grouped by session (sid), ranking each session's candidates by score_fn."""
    by_sid: dict[str, list] = defaultdict(list)
    for r in rows:
        by_sid[r["sid"]].append(r)
    rr = hits = 0.0
    for sid, rs in by_sid.items():
        ranked = sorted(rs, key=lambda r: -score_fn(r["f"]))
        for i, r in enumerate(ranked):
            if r["y"] == 1:
                rr += 1.0 / (i + 1)
                hits += 1.0 if i < 10 else 0.0
                break
    n = len(by_sid)
    return rr / n, hits / n


def main() -> None:
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    sids = sorted({r["sid"] for r in rows})
    # held-out split by SESSION (never split a session across train/test), leak-stratified via sid name
    test_sids = {s for i, s in enumerate(sids) if i % 5 == 0}
    train = [r for r in rows if r["sid"] not in test_sids]
    test = [r for r in rows if r["sid"] in test_sids]

    Xtr = np.array([r["f"] for r in train], dtype="float64")
    ytr = np.array([r["y"] for r in train], dtype="int64")
    mean, std = Xtr.mean(0), Xtr.std(0)
    std[std == 0] = 1.0
    Xs = (Xtr - mean) / std

    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
    clf.fit(Xs, ytr)
    w, b = clf.coef_[0], float(clf.intercept_[0])

    def model_score(f):
        return float(np.dot((np.array(f) - mean) / std, w) + b)

    def retrieval_score(f):
        return f[FEATURE_NAMES.index("retrieval_rank")]

    print("per-feature weight (standardized):")
    for name, wt in sorted(zip(FEATURE_NAMES, w), key=lambda x: -abs(x[1])):
        print(f"  {name:>16}: {wt:+.3f}")
    tr_mrr, tr_hit = session_metrics(train, model_score)
    te_mrr, te_hit = session_metrics(test, model_score)
    base_mrr, base_hit = session_metrics(test, retrieval_score)
    print(f"\nTRAIN  MRR {tr_mrr:.3f}  hit@10 {tr_hit:.3f}")
    print(f"TEST   MRR {te_mrr:.3f}  hit@10 {te_hit:.3f}   (held-out sessions)")
    print(f"TEST baseline (retrieval order)  MRR {base_mrr:.3f}  hit@10 {base_hit:.3f}")
    # per-leak held-out breakdown
    for leak in ("leaky", "moderate", "free"):
        sub = [r for r in test if r["leak"] == leak]
        if sub:
            m, h = session_metrics(sub, model_score)
            bm, bh = session_metrics(sub, retrieval_score)
            print(f"  {leak:>9}: model MRR {m:.3f}/hit {h:.3f}  vs retrieval {bm:.3f}/{bh:.3f}")

    Path(MODEL_OUT).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"features": FEATURE_NAMES, "weights": w.tolist(), "intercept": b,
               "mean": mean.tolist(), "std": std.tolist()},
              open(MODEL_OUT, "w"), indent=2)
    print(f"\nsaved {MODEL_OUT}")


if __name__ == "__main__":
    main()
