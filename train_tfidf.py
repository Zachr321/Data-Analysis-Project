"""
TF-IDF baselines: logistic regression and NB-SVM, one classifier per label.

Needs the outputs of preprocess.py (out/train_clean.csv, out/test_clean.csv, out/folds.csv).

Usage:
  python train_tfidf.py --data out --preds preds            # val on fold 0, then retrain on all train -> test
  python train_tfidf.py --data out --preds preds --val-only # quick: only the validation run
  python train_tfidf.py --data out --preds preds --test-only # only the final train -> test run

Writes to preds/:
  tfidf_lr_val.csv, tfidf_nbsvm_val.csv     predictions on the validation fold
  tfidf_lr_test.csv, tfidf_nbsvm_test.csv   predictions on the labeled test set
  nb_top_words.csv                          words with the highest NB ratio r per label (lexicon analysis)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from preprocess import LABELS, build_tfidf


# ------------------------------------------------------------------ NB-SVM
def nb_ratio(X, y, alpha: float = 1.0) -> np.ndarray:
    """r = log( (p / |p|_1) / (q / |q|_1) ),  p = alpha + sum of rows with y=1,  q = same for y=0."""
    p = alpha + np.asarray(X[y == 1].sum(axis=0)).ravel()
    q = alpha + np.asarray(X[y == 0].sum(axis=0)).ravel()
    return np.log((p / p.sum()) / (q / q.sum()))


class NBSVM:
    """Logistic regression on features scaled by the Naive Bayes log-count ratio r."""

    def __init__(self, C: float = 4.0):
        self.C = C

    def fit(self, X, y):
        self.r = sparse.diags(nb_ratio(X, y))
        self.clf = LogisticRegression(C=self.C, solver="liblinear", dual=True, max_iter=1000)
        self.clf.fit(X @ self.r, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X @ self.r)


MODELS = {
    "lr": lambda: LogisticRegression(C=4.0, solver="liblinear", max_iter=1000),
    "nbsvm": lambda: NBSVM(C=4.0),
}


# ------------------------------------------------------------------ helpers
def featurize(train_text, *other_texts):
    word, char = build_tfidf(train_text)                       # fit on TRAIN only
    to_X = lambda t: sparse.hstack([word.transform(t), char.transform(t)]).tocsr()
    return word, [to_X(train_text)] + [to_X(t) for t in other_texts]


def fit_predict(X_tr, Y_tr, X_ev, ids, out_path, model_name):
    pred = pd.DataFrame({"id": ids})
    models = {}
    for c in LABELS:
        m = MODELS[model_name]().fit(X_tr, Y_tr[c].values)
        pred[c] = m.predict_proba(X_ev)[:, 1]
        models[c] = m
    pred.to_csv(out_path, index=False)
    return models


def run(train_df, eval_df, tag, preds_dir):
    t0 = time.time()
    word, (X_tr, X_ev) = featurize(train_df.text_tfidf, eval_df.text_tfidf)
    print(f"[{tag}] features: {X_tr.shape[1]:,}  train rows: {X_tr.shape[0]:,}  ({time.time() - t0:.0f}s)")
    out = {}
    for name in MODELS:
        t0 = time.time()
        out[name] = fit_predict(X_tr, train_df, X_ev, eval_df.id, preds_dir / f"tfidf_{name}_{tag}.csv", name)
        print(f"[{tag}] {name} done ({time.time() - t0:.0f}s)")
    return word, out


def top_words(word_vec, nbsvm_models, k: int = 30) -> pd.DataFrame:
    """Highest-r WORD features per label (char n-grams skipped: not readable)."""
    names = word_vec.get_feature_names_out()
    n = len(names)
    cols = {}
    for c, m in nbsvm_models.items():
        r = m.r.diagonal()[:n]
        cols[c] = names[np.argsort(-r)[:k]]
    return pd.DataFrame(cols)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out")
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--test-only", action="store_true")
    a = ap.parse_args()
    data, preds = Path(a.data), Path(a.preds)
    preds.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data / "train_clean.csv").merge(pd.read_csv(data / "folds.csv"), on="id")
    for df in (train,):
        df["text_tfidf"] = df.text_tfidf.fillna("")

    # 1) validation run: train on folds != val, predict the val fold
    if not a.test_only:
        tr, va = train[train.fold != a.val_fold], train[train.fold == a.val_fold]
        word, models = run(tr, va, "val", preds)
        top_words(word, models["nbsvm"]).to_csv(preds / "nb_top_words.csv", index=False)

    if a.val_only:
        return

    # 2) final run: train on ALL train, predict the labeled test set
    test = pd.read_csv(data / "test_clean.csv")
    test["text_tfidf"] = test.text_tfidf.fillna("")
    run(train, test, "test", preds)


if __name__ == "__main__":
    main()
