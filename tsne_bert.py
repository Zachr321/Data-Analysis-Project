"""
t-SNE picture of how the fine-tuned BERT sees comments (lecture 7-1).

Takes a sample of labeled test comments, gets BERT's [CLS] sentence vector for each
(the vector the classifier head reads), squeezes it to 2D with t-SNE and colours each
point by its most specific label. Run it where the trained model is (GPU is faster,
CPU works for a few thousand comments).

Usage:
  python tsne_bert.py --model preds/bert_best --data out --n 4000 --out tsne_bert.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

from preprocess import LABELS

# one colour per comment: the most specific label it has (checked in this order)
GROUPS = [("threat", "threat"), ("identity_hate", "identity_hate"), ("severe_toxic", "severe_toxic"),
          ("insult", "insult"), ("obscene", "obscene (no insult)"), ("toxic", "toxic only")]
COLORS = {"clean": "#c9c9c9", "toxic only": "#f2b134", "obscene (no insult)": "#3b8bd0",
          "insult": "#e4572e", "severe_toxic": "#6a3d9a", "identity_hate": "#17a589", "threat": "#111111"}


def group_of(row) -> str:
    for col, name in GROUPS:
        if row[col] == 1:
            return name
    return "clean"


def sample(test: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Half clean, half labeled; every threat / identity_hate comment kept if it fits, so rare groups are visible."""
    rng = np.random.RandomState(seed)
    lab = test[test[LABELS].sum(axis=1) > 0]
    rare = lab[(lab.threat == 1) | (lab.identity_hate == 1)]
    rare = rare.sample(min(len(rare), n // 4), random_state=rng)
    other = lab.drop(rare.index)
    other = other.sample(min(len(other), n // 2 - len(rare)), random_state=rng)
    clean = test[test[LABELS].sum(axis=1) == 0].sample(n - len(rare) - len(other), random_state=rng)
    return pd.concat([rare, other, clean]).reset_index(drop=True)


@torch.no_grad()
def cls_vectors(texts, model_dir, max_len, batch, device):
    tok = AutoTokenizer.from_pretrained(model_dir)
    enc_model = AutoModel.from_pretrained(model_dir).to(device).eval()   # encoder only, no classifier head
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            h = enc_model(**enc).last_hidden_state[:, 0]                   # [CLS] vector
        out.append(h.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="preds/bert_best")
    ap.add_argument("--data", default="out")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tsne_bert.png")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = pd.read_csv(f"{a.data}/test_clean.csv")
    test["text_bert"] = test.text_bert.fillna("")
    s = sample(test, a.n, a.seed)
    s["group"] = s.apply(group_of, axis=1)
    print(s.group.value_counts().to_string())

    X = cls_vectors(s.text_bert.tolist(), a.model, a.max_len, a.batch, device)
    xy = TSNE(n_components=2, perplexity=a.perplexity, init="pca", random_state=a.seed).fit_transform(X)
    s[["x", "y"]] = xy
    s[["id", "group", "x", "y"]].to_csv(a.out.rsplit(".", 1)[0] + ".csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    for name in ["clean", "toxic only", "obscene (no insult)", "insult", "severe_toxic", "identity_hate", "threat"]:
        g = s[s.group == name]
        ax.scatter(g.x, g.y, s=6 if name == "clean" else 9, c=COLORS[name], label=f"{name} ({len(g)})",
                   alpha=0.55 if name == "clean" else 0.85, linewidths=0)
    ax.set_title("Fine-tuned BERT [CLS] vectors, t-SNE (labeled test sample)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2.5, fontsize=8, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
