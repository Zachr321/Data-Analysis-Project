"""
Shared evaluation for every model.

A prediction file is a CSV with columns: id, toxic, severe_toxic, obscene, threat, insult, identity_hate
(probabilities, same format as Kaggle's sample_submission).

Usage:
  python evaluate.py --pred preds/tfidf_lr_test.csv --truth out/test_clean.csv
  python evaluate.py --pred a.csv b.csv c.csv --truth out/test_clean.csv   # compare models

  # + yes/no metrics: cut-off per label chosen on validation, measured on test
  python evaluate.py --pred preds/tfidf_lr_test.csv --truth out/test_clean.csv \
                     --val-pred preds/tfidf_lr_val.csv --val-truth out/train_clean.csv

ROC-AUC needs no cut-off (it only looks at the ORDER of the probabilities).
Precision / recall / F1 / BCR need one: each label gets its own cut-off, chosen on the
validation fold (max F1) and then applied unchanged to the test set.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

MIN_POS = 10
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Subsets for the word-level vs sentence-level comparison (computed on the RAW comment text).
# 1) no_lexicon_word: comments without any common swear/insult word
LEX_RE = re.compile(
    r"fu+c+k|sh+i+t|bitch|cunt|dick|cock|\bass\b|asshole|fag|nigg|whore|slut|bastard|pussy|"
    r"penis|suck|crap|damn|piss|gay|stupid|idiot|moron|dumb|retard|loser|\bdie\b|kill|hate", re.I)
# 2) negated: an insult word with a negation earlier in the same clause ("I don't think you're stupid")
NEG = r"(?:not|no|never|n't|nobody|none|nothing|neither|nor|without)"
INS = r"(?:stupid|idiot|moron|dumb|retard\w*|loser|racist|liar|troll|vandal)"
NEG_RE = re.compile(rf"{NEG}\b[^.!?,;:\n]*?\b{INS}\b", re.I)


def subsets(raw_text: pd.Series) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(raw_text), bool),
        "no_lexicon_word": ~raw_text.str.contains(LEX_RE).values,
        "negated": raw_text.str.contains(NEG_RE).values,
    }


def auc_table(y: pd.DataFrame, p: pd.DataFrame, raw_text: pd.Series) -> pd.DataFrame:
    """Rows = subsets, columns = per-label ROC-AUC + mean. NaN if a label has < MIN_POS positives in a subset."""
    rows = {}
    for name, m in subsets(raw_text).items():
        r = {}
        for c in LABELS:
            yy, pp = y[c].values[m], p[c].values[m]
            # AUC needs both classes; with < MIN_POS positives it is too noisy to report
            r[c] = roc_auc_score(yy, pp) if MIN_POS <= yy.sum() < len(yy) else np.nan
        vals = [v for v in r.values() if not np.isnan(v)]
        r["mean"] = float(np.mean(vals)) if vals else np.nan
        r["n"] = int(m.sum())
        rows[name] = r
    return pd.DataFrame(rows).T[["n", "mean"] + LABELS]


def evaluate(pred_path, truth: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_csv(pred_path)
    df = truth[["id", "comment_text"] + LABELS].merge(p, on="id", suffixes=("", "_p"))
    if len(df) != len(p):
        print(f"warning: {len(p) - len(df)} predicted ids not found in truth")
    pred = df[[c + "_p" for c in LABELS]].set_axis(LABELS, axis=1)
    return auc_table(df[LABELS], pred, df.comment_text)


# ------------------------------------------------------------------ cut-off based metrics
CUTOFF_GRID = np.round(np.arange(0.02, 0.91, 0.01), 2)


def yes_no_metrics(y, p, cut):
    """Precision, recall, F1 and BCR at one cut-off.
    BCR = sqrt(TPR * TNR) (balanced classification rate, geometric-mean version)."""
    tn, fp, fn, tp = confusion_matrix(y, p >= cut, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "bcr": float(np.sqrt(rec * tnr)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def best_cutoffs(y_val: pd.DataFrame, p_val: pd.DataFrame) -> dict[str, float]:
    """Per label: the cut-off with the highest F1 on the validation fold."""
    out = {}
    for c in LABELS:
        f1s = [yes_no_metrics(y_val[c].values, p_val[c].values, t)["f1"] for t in CUTOFF_GRID]
        out[c] = float(CUTOFF_GRID[int(np.argmax(f1s))])
    return out


def _merge(pred_path, truth):
    p = pd.read_csv(pred_path)
    df = truth[["id"] + LABELS].merge(p, on="id", suffixes=("", "_p"))
    return df[LABELS], df[[c + "_p" for c in LABELS]].set_axis(LABELS, axis=1)


def cutoff_table(pred_path, truth, val_pred_path, val_truth) -> pd.DataFrame:
    """Test-set precision/recall/F1/BCR per label, at 0.5 and at the cut-off chosen on validation."""
    cuts = best_cutoffs(*_merge(val_pred_path, val_truth))
    y, p = _merge(pred_path, truth)
    rows = []
    for c in LABELS:
        for name, cut in (("0.5", 0.5), ("tuned", cuts[c])):
            m = yes_no_metrics(y[c].values, p[c].values, cut)
            rows.append({"label": c, "cutoff": f"{name} ({cut:.2f})", **m})
    t = pd.DataFrame(rows).set_index(["label", "cutoff"])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--truth", required=True, help="CSV with id, comment_text and the 6 label columns")
    ap.add_argument("--val-pred", nargs="+", help="validation predictions, same order as --pred (for cut-offs)")
    ap.add_argument("--val-truth", help="CSV with the validation labels (usually out/train_clean.csv)")
    a = ap.parse_args()
    truth = pd.read_csv(a.truth)
    summary = {}
    for path in a.pred:
        t = evaluate(path, truth)
        print(f"\n== {Path(path).stem}\n{t.round(4).to_string()}")
        summary[Path(path).stem] = t["mean"]
    if len(a.pred) > 1:
        print("\n== mean ROC-AUC by subset\n", pd.DataFrame(summary).round(4).to_string())

    if a.val_pred:
        vt = pd.read_csv(a.val_truth)
        for path, vpath in zip(a.pred, a.val_pred):
            t = cutoff_table(path, truth, vpath, vt)
            print(f"\n== {Path(path).stem}: test metrics at 0.5 vs cut-off chosen on validation")
            print(t.round(3).to_string())


if __name__ == "__main__":
    main()
