"""
Rank-average ensemble of the four models, weights and cut-offs chosen per label on validation fold 0.

1) Ranks. Each model's score for one label is turned into a percentile rank between 0 and 1, so a
   model with shifted probabilities (BERT + pos_weight) cannot dominate the average.
   Validation: rank among the validation comments.
   Test: rank against the SAME validation scores ("what share of validation comments scored lower").
   This puts test on the validation scale, so a cut-off chosen on validation still means the same thing.
2) Weights. For every label, all weight combinations in steps of 0.1 (summing to 1) are tried on
   fold 0 and the one with the best ROC-AUC is kept.
3) Cut-offs. Ensemble scores are ranks, so almost all positives sit above 0.9 and the 0.02-0.90 grid
   in evaluate.py is useless here. Instead we try cut-offs at the 80.0-99.9th percentiles of the
   validation ensemble score and keep the one with the best F1.

Usage:  python ensemble.py --preds preds --data out
Writes: preds/ensemble_val.csv, preds/ensemble_test.csv, preds/ensemble_weights.json,
        preds/ensemble_cutoffs.json, results/ensemble_test_metrics.csv
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from evaluate import yes_no_metrics
from preprocess import LABELS

MODELS = ["tfidf_lr", "tfidf_nbsvm", "bert", "bert_pw"]
CUT_QUANTILES = np.linspace(0.80, 0.999, 400)


def load(preds, split, ids):
    return {m: pd.read_csv(preds / f"{m}_{split}.csv").set_index("id").loc[ids] for m in MODELS}


def pct_rank(x):
    """Percentile rank of each value inside its own array (0 = lowest, 1 = highest)."""
    return (rankdata(x) - 1) / (len(x) - 1)


def rank_against(x, ref):
    """Share of the reference values that are lower than each x (ties count half)."""
    ref = np.sort(ref)
    lo = np.searchsorted(ref, x, side="left")
    hi = np.searchsorted(ref, x, side="right")
    return (lo + hi) / 2 / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--data", default="out")
    ap.add_argument("--results", default="results")
    ap.add_argument("--step", type=float, default=0.1)
    a = ap.parse_args()
    preds, data, results = Path(a.preds), Path(a.data), Path(a.results)
    results.mkdir(exist_ok=True)

    val_ids = pd.read_csv(preds / f"{MODELS[2]}_val.csv").id
    test_ids = pd.read_csv(preds / f"{MODELS[2]}_test.csv").id
    yv = pd.read_csv(data / "train_clean.csv", usecols=["id", *LABELS]).set_index("id").loc[val_ids]
    yt = pd.read_csv(data / "test_clean.csv", usecols=["id", *LABELS]).set_index("id").loc[test_ids]
    pv, pt = load(preds, "val", val_ids), load(preds, "test", test_ids)

    n = int(round(1 / a.step))
    grid = [np.array(c) / n for c in itertools.product(range(n + 1), repeat=len(MODELS)) if sum(c) == n]

    weights, cutoffs, rows = {}, {}, []
    ens_v, ens_t = pd.DataFrame({"id": val_ids}), pd.DataFrame({"id": test_ids})
    for c in LABELS:
        Rv = np.column_stack([pct_rank(pv[m][c].values) for m in MODELS])
        Rt = np.column_stack([rank_against(pt[m][c].values, pv[m][c].values) for m in MODELS])
        aucs = [roc_auc_score(yv[c], Rv @ w) for w in grid]
        w = grid[int(np.argmax(aucs))]
        sv, st = Rv @ w, Rt @ w
        weights[c] = dict(zip(MODELS, w.round(2).tolist()))
        ens_v[c], ens_t[c] = sv, st

        cands = np.unique(np.quantile(sv, CUT_QUANTILES))
        f1s = [yes_no_metrics(yv[c].values, sv, t)["f1"] for t in cands]
        cut = float(cands[int(np.argmax(f1s))])
        cutoffs[c] = cut
        m = yes_no_metrics(yt[c].values, st, cut)
        rows.append({"label": c, **weights[c], "val_auc": max(aucs), "test_auc": roc_auc_score(yt[c], st),
                     "cutoff": cut, **{k: m[k] for k in ("precision", "recall", "f1", "bcr")}})

    t = pd.DataFrame(rows).set_index("label")
    t.loc["mean", ["val_auc", "test_auc", "precision", "recall", "f1", "bcr"]] = \
        t[["val_auc", "test_auc", "precision", "recall", "f1", "bcr"]].mean()
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(na_rep=""))

    ens_v.to_csv(preds / "ensemble_val.csv", index=False)
    ens_t.to_csv(preds / "ensemble_test.csv", index=False)
    json.dump(weights, open(preds / "ensemble_weights.json", "w"), indent=1)
    json.dump({c: round(v, 4) for c, v in cutoffs.items()}, open(preds / "ensemble_cutoffs.json", "w"), indent=1)
    t.round(4).to_csv(results / "ensemble_test_metrics.csv")


if __name__ == "__main__":
    main()
