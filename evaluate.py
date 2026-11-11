"""
Shared evaluation for every model.

A prediction file is a CSV with columns: id, toxic, severe_toxic, obscene, threat, insult, identity_hate
(probabilities, same format as Kaggle's sample_submission).

Usage:
  python evaluate.py --pred preds/tfidf_lr_test.csv --truth out/test_clean.csv
  python evaluate.py --pred a.csv b.csv c.csv --truth out/test_clean.csv   # compare models
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--truth", required=True, help="CSV with id, comment_text and the 6 label columns")
    a = ap.parse_args()
    truth = pd.read_csv(a.truth)
    summary = {}
    for path in a.pred:
        t = evaluate(path, truth)
        print(f"\n== {Path(path).stem}\n{t.round(4).to_string()}")
        summary[Path(path).stem] = t["mean"]
    if len(a.pred) > 1:
        print("\n== mean ROC-AUC by subset\n", pd.DataFrame(summary).round(4).to_string())


if __name__ == "__main__":
    main()
