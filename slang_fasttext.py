"""
Slang detection with fastText (future-work proof of concept).

Idea: words used like known cuss words end up with vectors close to them. Starting from a
public profanity list (the anchors), any word that is very similar to the anchors but not on
the list is a candidate new cuss word / slang.

  1. Train fastText (skip-gram, 10 epochs, word pieces of 4-6 letters) on the Jigsaw training
     comments (min_count >= 5). --kind word2vec trains plain word2vec instead (no word pieces).
  2. Anchors = public profanity list  ∩  our vocabulary  minus neutral identity terms, keeping only
     words with r > 0 here (listed words that are not more common in toxic comments are dropped).
  3. Leave-out test: hide 30% of the anchors, score every word by its average cosine similarity
     to its 3 most similar visible anchors, and check how well the hidden anchors are separated
     from a random, frequency-matched sample of normal words (recall, false positive rate, AUC).
     The threshold is chosen here (best recall - FPR), averaged over 5 random splits.
  4. Candidates = words with score >= threshold that are not on the list.
  5. (Each time new text arrives) update() continues training on ONLY the new text and returns
     candidates that appear in it. Demo: the test comments play the role of "new text".

Usage:
  python slang_fasttext.py --data out --profanity ldnoobw_en.txt --out slang
  (ldnoobw_en.txt = https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words, file "en")

Writes to slang/: <kind>.model, anchors.csv, leaveout.csv, candidates.csv, candidates_new_text.csv
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from sklearn.metrics import roc_auc_score, roc_curve
from wordfreq import top_n_list

from preprocess import LABELS, SWEAR_STEMS

# group names are targets, not profanity: never use them as anchors (bias)
IDENTITY_TERMS = {"gay", "gays", "lesbian", "jew", "jews", "jewish", "muslim", "muslims", "black", "blacks",
                  "white", "queer", "homosexual", "transgender", "negro", "arab", "asian", "mexican"}
TOP_K = 3            # score = mean similarity to the TOP_K most similar anchors (not just one)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'*@$!_]*[a-z0-9*]|[a-z0-9]", text.lower())


def nb_ratio(docs, y):
    """Document-level r(w) = log P(w | toxic) / P(w | clean), add-1 smoothed."""
    a, b = Counter(), Counter()
    for d, l in zip(docs, y):
        (a if l else b).update(set(d))
    n1, n0 = y.sum(), (~y).sum()
    return {w: np.log(((a[w] + 1) / (n1 + 2)) / ((b[w] + 1) / (n0 + 2))) for w in set(a) | set(b)}


def scores(model, words, anchors):
    """Mean cosine similarity of each word to its TOP_K most similar anchors."""
    W = model.wv[words]
    A = model.wv[anchors]
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    S = W @ A.T
    k = min(TOP_K, S.shape[1])
    return np.sort(S, axis=1)[:, -k:].mean(axis=1)


def normal_sample(vocab_freq, exclude, r, like_words, n, rng):
    """Random normal words, frequency-matched to `like_words`.
    Normal = not on the list, alphabetic, and
      - no swear stem and r <= 1, or
      - contains a swear stem BUT is a real English word that is not toxic here (r <= 0):
        cocktail, dickens, assume, passion. These are kept on purpose: they are exactly the
        words where fastText's letter pieces can fool the detector.
    Cuss variants like fuckin / dickhead (stem + toxic here) are never counted as normal."""
    english = set(top_n_list("en", 100_000))
    def is_normal(w):
        if w in exclude or not w.isalpha():
            return False
        if any(s in w for s in SWEAR_STEMS):
            return w in english and r.get(w, 0) <= 0
        return r.get(w, 0) <= 1
    pool = [w for w in vocab_freq if is_normal(w)]
    bins = lambda f: int(np.log2(f))
    by_bin = {}
    for w in pool:
        by_bin.setdefault(bins(vocab_freq[w]), []).append(w)
    target = Counter(bins(vocab_freq[w]) for w in like_words)
    out = []
    per_word = max(1, n // max(1, len(like_words)))
    for b_, cnt in target.items():
        cand = by_bin.get(b_, [])
        take = min(len(cand), cnt * per_word)
        out += list(rng.choice(cand, take, replace=False)) if take else []
    # always include every stem-containing normal word (the hard cases), not just a random share
    stem_words = [w for w in pool if any(s in w for s in SWEAR_STEMS)]
    return list(dict.fromkeys(out + stem_words))


def leave_out(model, anchors, vocab_freq, known, r, splits=5, hide=0.3, n_normal=400, seed=42):
    rng = np.random.RandomState(seed)
    rows, curves = [], []
    for s in range(splits):
        a = np.array(anchors)
        rng.shuffle(a)
        n_hide = int(len(a) * hide)
        hidden, visible = list(a[:n_hide]), list(a[n_hide:])
        normal = normal_sample(vocab_freq, known, r, hidden, n_normal, rng)
        sh, sn = scores(model, hidden, visible), scores(model, normal, visible)
        y = np.r_[np.ones(len(sh)), np.zeros(len(sn))]
        sc = np.r_[sh, sn]
        fpr, tpr, thr = roc_curve(y, sc)
        best = int(np.argmax(tpr - fpr))
        rows.append({"split": s, "hidden": len(sh), "normal": len(sn), "auc": roc_auc_score(y, sc),
                     "threshold": thr[best], "recall": tpr[best], "fpr": fpr[best]})
        curves.append((sh, sn))
    return pd.DataFrame(rows), curves


def candidates(model, words, anchors, threshold, known):
    words = [w for w in words if w not in known]
    sc = scores(model, words, anchors)
    df = pd.DataFrame({"word": words, "score": sc})
    df["has_swear_stem"] = df.word.apply(lambda w: any(s in w for s in SWEAR_STEMS))
    return df[df.score >= threshold].sort_values("score", ascending=False)


def update(model, new_sentences, epochs=5):
    """Future text: add its words to the vocabulary and train on ONLY the new text."""
    model.build_vocab(new_sentences, update=True)
    model.train(new_sentences, total_examples=len(new_sentences), epochs=epochs)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out")
    ap.add_argument("--profanity", default="ldnoobw_en.txt")
    ap.add_argument("--out", default="slang")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--kind", choices=["fasttext", "word2vec"], default="fasttext")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(Path(a.data) / "train_clean.csv")
    docs = [tokenize(t) for t in train.text_bert.fillna("")]
    y = (train[LABELS].sum(axis=1) > 0).values

    # 1) fastText on the training comments
    # skip-gram (sg=1) learns from context much better than the default CBOW; with CBOW and short
    # word pieces the vectors were dominated by spelling (leave-out AUC 0.66 vs 0.96)
    if a.kind == "fasttext":
        model = FastText(docs, vector_size=a.dim, window=5, min_count=a.min_count, sg=1, min_n=4, max_n=6,
                         epochs=a.epochs, workers=2, seed=a.seed)
    else:
        model = Word2Vec(docs, vector_size=a.dim, window=5, min_count=a.min_count, sg=1,
                         epochs=a.epochs, workers=2, seed=a.seed)
    model.save(str(out / f"{a.kind}.model"))
    freq = {w: model.wv.get_vecattr(w, "count") for w in model.wv.index_to_key}
    print(f"fastText vocabulary (min_count {a.min_count}): {len(freq):,} words")

    # 2) anchors: public list ∩ vocabulary − identity terms (+ r check)
    plist = {l.strip().lower() for l in open(a.profanity, encoding="utf-8") if l.strip() and " " not in l.strip()}
    r = nb_ratio(docs, y)
    in_vocab = sorted(w for w in plist if w in freq and w not in IDENTITY_TERMS)
    adf = pd.DataFrame({"anchor": in_vocab, "count": [freq[w] for w in in_vocab], "r": [r.get(w, 0) for w in in_vocab]})
    adf["used"] = adf.r > 0
    adf.sort_values("r").to_csv(out / "anchors.csv", index=False)
    anchors = adf[adf.used].anchor.tolist()
    known = plist | set(in_vocab)
    print(f"anchors: {len(in_vocab)} of {len(plist)} list words are in our vocabulary; dropped "
          f"{(~adf.used).sum()} with r <= 0 (listed, but not toxic here): {', '.join(adf[~adf.used].anchor)}; "
          f"using {len(anchors)}")

    # 3) leave-out test
    lo, curves = leave_out(model, anchors, freq, known, r, seed=a.seed)
    lo.to_csv(out / "leaveout.csv", index=False)
    print("\nleave-out test (hide 30% of anchors, 5 splits):")
    print(lo.round(3).to_string(index=False))
    m, sd = lo.mean(), lo.std()
    threshold = float(m.threshold)
    print(f"mean AUC {m.auc:.3f} ± {sd.auc:.3f} | threshold {threshold:.3f} -> "
          f"recall {m.recall:.2f} ± {sd.recall:.2f}, FPR {m.fpr:.2f} ± {sd.fpr:.2f}")

    traps = [w for w in ["cocktail", "dickens", "assassin", "classic", "cockpit", "peacock", "hancock",
                         "dickinson", "scunthorpe", "therapist", "shitake", "assume", "passion"] if w in freq]
    if traps:
        ts = scores(model, traps, anchors)
        flagged = [w for w, v in zip(traps, ts) if v >= threshold]
        print(f"spelling traps flagged: {len(flagged)}/{len(traps)} -> {', '.join(flagged) or 'none'}")

    # 4) candidates in our own data
    cand = candidates(model, list(freq), anchors, threshold, known)
    cand["count"] = cand.word.map(freq)
    cand["r"] = cand.word.map(lambda w: r.get(w, 0))
    cand.to_csv(out / "candidates.csv", index=False)
    print(f"\ncandidates in training data: {len(cand)} words (top, no swear stem):")
    print(cand[~cand.has_swear_stem].head(25)[["word", "score", "count", "r"]].round(3).to_string(index=False))

    # 5) demo of the future-work loop: test comments play the role of "new text"
    test = pd.read_csv(Path(a.data) / "test_clean.csv")
    new_docs = [tokenize(t) for t in test.text_bert.fillna("")]
    model = update(model, new_docs, epochs=a.epochs)
    new_counts = Counter(w for d in new_docs for w in d)
    new_words = [w for w, c in new_counts.items() if c >= a.min_count and w in model.wv.key_to_index]
    ncand = candidates(model, new_words, anchors, threshold, known)
    ncand["new_text_count"] = ncand.word.map(new_counts)
    ncand["seen_before"] = ncand.word.isin(freq)
    ncand.to_csv(out / "candidates_new_text.csv", index=False)
    print(f"\nafter update on new text: {len(ncand)} candidates used >= {a.min_count}x in it, "
          f"{(~ncand.seen_before).sum()} of them never seen in training")
    print(ncand[~ncand.has_swear_stem].head(15)[["word", "score", "new_text_count", "seen_before"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
