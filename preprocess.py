"""
Preprocessing for the Jigsaw toxic comment data.

Two outputs per comment:
  text_tfidf : full pipeline, for TF-IDF + lexicon / NB-SVM
  text_bert  : light pipeline, for an uncased BERT

Run:
  python preprocess.py --out out/
"""
import argparse
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import OSA          # edit distance; swapping two neighbors counts as 1
from wordfreq import top_n_list

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ---------------------------------------------------------------- config
# Target words for deobfuscation. A disguised token can only become one of these.
LEXICON = {
    "fuck", "fucking", "fucker", "motherfucker", "shit", "bitch", "cunt", "dick",
    "cock", "asshole", "bastard", "faggot", "whore", "slut", "pussy", "retard",
    "idiot", "stupid", "nigger",
}
# English guard for the edit-distance step: 100k most common words, MINUS anything that
# contains a swear stem. wordfreq is built from web/social text, so it also lists slang
# spellings like "fuckin", "biatch", "fuuuuck"; those must not be protected.
SWEAR_STEMS = ("fuc", "fuk", "shit", "bitch", "biatch", "cunt", "dick", "cock", "asshol",
               "bastard", "fag", "whor", "slut", "pussy", "retard", "idiot", "stupid", "nigg", "niga")
ENGLISH = {w for w in top_n_list("en", 100_000) if not any(s in w for s in SWEAR_STEMS)}

LEET = str.maketrans({"1": "i", "!": "i", "0": "o", "3": "e", "4": "a",
                      "@": "a", "$": "s", "5": "s", "7": "t"})   # no '*': it's a wildcard
MASK_CHARS = "*#%"                           # each one = exactly one unknown letter
CHAT = {"u": "you", "ur": "your", "r": "are", "urself": "yourself", "ure": "you're"}
MAX_TOKEN_REPEATS = 5                        # spam cap: keep at most 5 copies of a token per comment

# ---------------------------------------------------------------- step 1: wikipedia noise
RE_TIME = re.compile(r"\d{1,2}:\d{2},? \d{1,2} \w+ \d{4} \(utc\)", re.I)
RE_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
RE_USER = re.compile(r"\[\[\s*user(?:[ _]talk)?:[^\]|]*(?:\|[^\]]*)?\]\]", re.I)
RE_MARKUP = re.compile(r"\[\[|\]\]|\{\{|\}\}|={2,}|'{2,}")


def replace_noise(t: str) -> str:
    t = RE_TIME.sub(" timetoken ", t)
    t = RE_IP.sub(" iptoken ", t)
    t = RE_URL.sub(" urltoken ", t)
    t = RE_USER.sub(" usertoken ", t)
    t = RE_MARKUP.sub(" ", t)
    return t


# ---------------------------------------------------------------- step 4: deobfuscation
def _collapse(t: str, keep: int) -> str:
    """Shorten runs of 3+ identical letters to `keep` letters."""
    return re.sub(r"([a-z])\1{2,}", r"\1" * keep, t)


@lru_cache(maxsize=None)
def deobfuscate(tok: str) -> tuple[str, str]:
    """Return (new_text, rule). rule == '' means no lexicon change was made.
    Rules a-d only fire if the result is EXACTLY a lexicon word, so they are safe.
    Rule e is a fuzzy guess: it keeps the original and adds '<word>_guess'."""
    if tok in LEXICON or not re.search(r"[a-z]", tok):
        return tok, ""
    base = _collapse(tok, 2)                  # text rule for everything: goooood -> good

    # a) separators, only f.u.c.k style (every piece 1-2 chars); re-add, u.s. stay untouched
    t = tok
    parts = re.split(r"[.\-_]", t)
    if len(parts) > 1 and all(0 < len(p) <= 2 for p in parts):
        t = "".join(parts)

    # b) repeated letters: try 2 (asssshole -> asshole) and 1 (fuckkk -> fuck)
    t2 = _collapse(t, 2)
    for cand in (t2, _collapse(t, 1)):
        if cand in LEXICON:
            return cand, "repeat/separator"

    # c) wildcard: * # % = one unknown letter; replace only if exactly one lexicon word fits
    if any(c in t2 for c in MASK_CHARS):
        pat = re.compile("^" + "".join("[a-z]" if c in MASK_CHARS else re.escape(c) for c in t2) + "$")
        hits = [w for w in LEXICON if pat.match(w)]
        if len(hits) == 1:
            return hits[0], "wildcard"

    # d) leet + sound-alike (ph -> f)
    t3 = t2.translate(LEET).replace("ph", "f")
    for cand in (t3, _collapse(t3, 1)):
        if cand in LEXICON:
            return cand, "leet"

    # inflected forms that already contain the whole word (dicks, fucked, idiots) are real
    # words, not disguises: leave them alone (word + char n-grams already link them)
    if any(t3.startswith(w) for w in LEXICON):
        return base, ""

    # e) edit distance = 1, same first letter, >= 4 letters, not an English word -> add a guess
    if len(t3) >= 4 and t3.isalpha() and t3 not in ENGLISH and base not in ENGLISH:
        best = min(LEXICON, key=lambda w: OSA.distance(t3, w))
        if OSA.distance(t3, best) == 1 and t3[0] == best[0]:
            return f"{base} {best}_guess", "edit_distance"

    return base, ""


# ---------------------------------------------------------------- pipeline
RE_SPLIT = re.compile(r"^([\"'(\[<{]*)(.*?)([.,;:?!\"')\]>}]*)$")


def tokenize(t: str) -> list[str]:
    """Split on whitespace, peel quotes/brackets/punctuation off the ends.
    Symbols INSIDE a word (sh!t, f**k, a$$hole) are kept for deobfuscation.
    Trailing ! and ? are kept as their own token (once)."""
    toks = []
    for raw in t.split():
        _, core, trail = RE_SPLIT.match(raw).groups()
        if core:
            toks.append(core)
        for p in "!?":
            if p in trail:
                toks.append(p)
    return toks


def cap_repeats(toks: list[str], k: int = MAX_TOKEN_REPEATS) -> list[str]:
    seen, out = Counter(), []
    for tok in toks:
        seen[tok] += 1
        if seen[tok] <= k:
            out.append(tok)
    return out


def clean(text: str, mode: str = "tfidf", log: list | None = None) -> str:
    """mode='tfidf': full pipeline. mode='bert': noise + deobfuscation + spam cap only."""
    t = replace_noise(text)                                  # 1 noise tokens
    t = t.lower()                                            # 2 lowercase
    toks = tokenize(t)
    if mode == "tfidf":
        toks = [CHAT.get(w, w) for w in toks]                # 3 chat spelling
    out = []
    for w in toks:                                           # 4 deobfuscation
        new, rule = deobfuscate(w)
        if rule and log is not None:
            log.append((w, new, rule))
        out.extend(new.split())
    out = cap_repeats(out)                                   # 5 spam cap
    return " ".join(out)


# ---------------------------------------------------------------- data helpers
def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate comments; if duplicates disagree, keep the max label (any rater said yes)."""
    key = df.comment_text.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    agg = {c: "max" for c in LABELS} | {"id": "first", "comment_text": "first"}
    return df.assign(_k=key).groupby("_k", as_index=False).agg(agg)[df.columns]


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.Series:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    fold = pd.Series(-1, index=df.index)
    skf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (_, va) in enumerate(skf.split(df, df[LABELS].values)):
        fold.iloc[va] = i
    return fold


def build_tfidf(train_text, max_features: int = 200_000):
    """Fit on TRAIN only. Returns (word_vec, char_vec); call .transform on val/test."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features,
                           sublinear_tf=True, token_pattern=r"\S+")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=max_features, sublinear_tf=True)
    word.fit(train_text)
    char.fit(train_text)
    return word, char


def input_csv(data_dir: Path, filename: str) -> Path:
    """Find a dataset CSV stored flat or in the repository's same-name folder."""
    candidates = (data_dir / filename, data_dir / filename / filename)
    for path in candidates:
        if path.is_file():
            return path
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}; expected {expected}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(input_csv(data, "train.csv"))
    n0 = len(train)
    train = dedupe(train)
    print(f"train: {n0} rows -> {len(train)} after dedupe")

    test = pd.read_csv(input_csv(data, "test.csv")).merge(
        pd.read_csv(input_csv(data, "test_labels.csv")), on="id"
    )
    test = test[test.toxic != -1].reset_index(drop=True)      # -1 = not scored
    print(f"test: {len(test)} labeled rows")

    log = []
    for df, name in [(train, "train"), (test, "test")]:
        df["text_tfidf"] = [clean(t, "tfidf", log) for t in df.comment_text]
        df["text_bert"] = [clean(t, "bert") for t in df.comment_text]
        df.to_csv(out / f"{name}_clean.csv", index=False)

    logdf = pd.DataFrame(log, columns=["original", "new", "rule"])
    (logdf.value_counts().rename("count").reset_index()
          .sort_values(["rule", "count"], ascending=[True, False])
          .to_csv(out / "deobf_log.csv", index=False))
    print("deobfuscation changes by rule:\n", logdf.rule.value_counts().to_string())

    train[["id"]].assign(fold=make_folds(train)).to_csv(out / "folds.csv", index=False)
    print("done ->", out)


if __name__ == "__main__":
    main()
