# Future work: detecting new slang and cuss words

Our classifiers (TF-IDF + LR / NB-SVM, BERT) only know insults that appear in their labeled training data. Online language keeps changing: an existing word can start being used as an insult, or a completely new one can appear. This document describes a lightweight **slang monitor** that finds such words in new, unlabeled text, and reports a proof-of-concept run on the Jigsaw data.

Code: [`slang_fasttext.py`](../slang_fasttext.py). Results: [`results/`](../results).

---

## 1. Idea

A word's embedding is learned from the words around it. If people start writing a word in the places where they would otherwise write `idiot` or `fuck`, its vector moves close to the vectors of those words. So:

1. Take a set of **known cuss words** (the *anchors*).
2. Learn word vectors from text.
3. Any word that is **very similar to the anchors** but **not yet on the known list** is a candidate new cuss word.

No labels are needed for the new text; the labeled Jigsaw data is only used to choose and check the anchors.

## 2. How it connects to the classifier

The monitor does **not** feed vectors into BERT. Its output is a **list of words**. The classifier can only learn a new insult from **labeled examples**, so the link goes through data:

```
new text ──► slang monitor ──► candidate words ──► human review ──► confirmed words
                                                                      │
              comments containing confirmed words  ◄──────────────────┘
                         │
                         ▼
              label them (by hand, or pseudo-labels) ──► fine-tune BERT
```

Confirmed words can also be added to the deobfuscation lexicon in `preprocess.py`.

## 3. Design decisions

### 3.1 Which embedding model: fastText (skip-gram)

| Option | Problem for this task |
|---|---|
| Pretrained word2vec / GloVe | Fixed snapshots (around 2013–2014): no recent slang, swear words are rare in news/Wikipedia text, and the released Google word2vec cannot be trained further |
| GloVe trained on our data | Learns from one co-occurrence table of the whole corpus: adding new text means rebuilding and retraining everything |
| BERT word vectors | Fixed vocabulary (new words are split into pieces), vectors need extra normalization before a similarity threshold means anything, GPU pass over all new text each time |
| **word2vec / fastText trained on our data** | Can **continue training on only the new text** (`build_vocab(update=True)` + `train()`) |

Between word2vec and fastText we chose **fastText**: it builds each word's vector from the word itself plus its character pieces, so rare and misspelled slang (`azzholes`, `fckin`) still gets a sensible vector.

The settings matter a lot:

| Setting | Leave-out ROC-AUC |
|---|---|
| fastText default (CBOW, pieces from 3 letters, 5 epochs) | 0.66 |
| **fastText skip-gram, pieces of 4–6 letters, 10 epochs** | **0.96** |
| word2vec skip-gram, 10 epochs (no pieces) | 0.94 |

With the default settings, short character pieces dominated the vectors, so words were "similar" mostly because they were spelled alike. Skip-gram learns more from context and handles rare words better. (These three numbers come from the first version of the test, see 3.5.)

Final settings: `sg=1, vector_size=100, window=5, min_count=5, min_n=4, max_n=6, epochs=10`, trained on the Jigsaw training comments (`text_bert` column, 42,729 words with at least 5 occurrences). `min_count=5` matters: a word seen only once or twice gets an unreliable vector.

### 3.2 Anchors: which words count as known cuss words

1. Start from a public profanity list: [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words), English file, single words only (279 entries).
2. Keep only words that appear in our comments at least 5 times (127 words). A word without a vector cannot be an anchor.
3. Remove **neutral identity terms** (`gay`, `jew`, `muslim`, …). They name a target group, not profanity; using them as anchors would teach the monitor that the group itself is offensive.
4. Keep only words with **$r > 0$**, where

   $$r(w) = \log \frac{P(w \mid \text{toxic comment})}{P(w \mid \text{clean comment})}$$

   is the same NB ratio NB-SVM uses (computed on the training labels, add-1 smoothed). This drops listed words that are not actually more common in toxic comments here: `bondage, erotic, escort, nambla, santorum`.

Result: **122 anchors** (`results/slang_anchors.csv`).

Why a public list and not our own data-driven $r$ list: the public list is independent of our labels, so the test cannot be accused of only rediscovering what our labels already contained. The $r$ values are used only as a check.

### 3.3 Scoring a word

For a word $w$, compute the cosine similarity to every anchor and average the **3 highest**:

$$\text{score}(w) = \frac{1}{3}\sum_{a \,\in\, \text{top-3 anchors}} \cos(v_w, v_a)$$

Why not "highly similar to one anchor is enough": embedding similarity means "used in similar contexts", not "means the same". `fuck` is close to `off` and `shut`, `dick` is close to names, opposites like `love`/`hate` are close. Averaging over three anchors makes one lucky match much less likely to flag a normal word.

### 3.4 Candidate rule

$$\text{candidate}(w) = \big(\text{score}(w) \ge \text{threshold}\big) \;\land\; \big(w \notin \text{known list}\big)$$

This covers brand-new words, existing words that are now used as insults, and cuss words the list simply missed. Only the **current** similarity is needed; there is no "old vs new model" comparison.

### 3.5 Choosing the threshold: the leave-out test

There is no list of "future slang" to test against, so we simulate it with words whose answer we know:

1. Randomly **hide 30% of the anchors** (36 words). They play the role of new slang: the monitor is never told they are cuss words.
2. Score every word against the remaining **visible** anchors.
3. Build a group of **normal words** (about 400, see below).
4. Measure how well the scores separate hidden cuss words from normal words:
   - **recall**: share of hidden cuss words with score ≥ threshold
   - **false positive rate (FPR)**: share of normal words with score ≥ threshold
   - **ROC-AUC** over all thresholds (probability that a random hidden cuss word scores higher than a random normal word)
5. Pick the threshold that maximizes recall − FPR, and **repeat over 5 random splits**.

Recall alone is not enough: a threshold of 0 catches every hidden word but also flags the whole vocabulary. That is why the normal group is needed.

**Building the normal group.** Normal words are drawn at random (never hand-picked, which would make the test too easy) and **frequency-matched** to the hidden words, because frequent words have more stable vectors. A word counts as normal if it is not on the list and either

- contains no swear stem and has $r \le 1$, or
- contains a swear stem (`cock`, `dick`, `ass`, …) **but is a real English word that is not toxic here** ($r \le 0$): `cockpit`, `dickens`, `peacock`.

Cuss variants that are not on the list (`fuckin`, `dickhead`: stem + toxic here) are never counted as normal, otherwise the monitor would be punished for scoring them high.

**Why the $r \le 1$ filter on words without a stem.** Many cuss words are neither on the list nor contain a stem: `wanker`, `dumbass`, `douchebag`, `moron`, `prick`, `scumbag`, `imbecile`. Without the filter they would count as "normal", and the monitor would be punished for correctly scoring them high. $r \le 1$ (at most about 2.7× more common in toxic comments) removes them while keeping ordinary words that are only slightly more common in toxic comments, such as `you` ($r = 0.32$) or `people` ($0.11$). The exact cut-off matters little:

| Filter on non-stem normal words | Leave-out ROC-AUC |
|---|---|
| none | 0.935 ± 0.013 |
| $r \le 2$ | 0.952 ± 0.013 |
| **$r \le 1$** (used) | **0.958 ± 0.010** |
| $r \le 0.5$ | 0.948 ± 0.010 |
| $r \le 0$ | 0.956 ± 0.016 |

Every value between 0 and 2 gives about the same result; only dropping the filter lowers the AUC, because real cuss words are then counted as false alarms.

The first version of the test excluded *every* word with a swear stem from the normal group. That also removed innocent words like `cockpit` and `dickens`, exactly the words where fastText's letter pieces can fool it, so the first numbers were too optimistic. The table below uses the fixed version.

## 4. Results (proof of concept on Jigsaw)

**Leave-out test** (fixed normal group, 5 splits; `results/slang_leaveout.csv`):

| | Value |
|---|---|
| ROC-AUC | **0.958 ± 0.010** |
| Best threshold per split | 0.51 – 0.59 |

| Threshold | Hidden cuss words caught (recall) | Normal words passing (FPR) |
|---|---|---|
| 0.545 (average of the per-split best) | 89% | 11% |
| **0.58** | **80%** | **4.6%** |

- The ranking is good (AUC ≈ 0.96), but the best threshold **varies a lot between splits**, so a fixed cut-off alone cannot be trusted.
- **Spelling still leaks.** Only 5 innocent English words containing a swear stem occur often enough to be tested, and 4 of them were flagged: `cockpit` (0.60), `dickens` (0.59), `cockney` (0.57), `dickinson` (0.55). All four have $r < 0$ (more common in clean comments). Plain word2vec flagged fewer such words in our comparison, but ranked hidden cuss words worse overall.

**Candidates in our own training data** (threshold 0.58, `results/slang_candidates_top.csv`): 4,935 words pass, 238 of them with score ≥ 0.75. The top of the ranking contains real cuss words that are **not on the public list**: `wanker`, `asshat`, `nobhead`, `arsewipe`, `butthead`, `buttface`, and slurs such as `wog` and `limey`. Because 4–11% of a 42k-word vocabulary passes the threshold, the full list is mostly noise; the **top of the ranking** is where the useful finds are.

**Simulated "new text"** (`results/slang_candidates_new_text_top.csv`): after training on the training comments, the model was updated with **only** the test comments, as if they were newly collected text. 1,748 candidates appear at least 5 times in the new text, and 598 of them were never seen during training, e.g. `wankest`, `ashole`, `azzholes`, `buttholes`, `dik`, `boaby`. The list also contains `asexual` and `pansexual`: the public list's anchors are mostly sexual terms, so sexuality words drift toward them. Flagging identity words as cuss words would be harmful; this is a bias risk of the anchor choice.

## 5. Human review (final decision)

Because of the unstable threshold, the spelling traps and the bias risk, the monitor should **rank**, and a person should **decide**:

1. The monitor sorts candidates by score.
2. A reviewer checks the **top 50–100**.
3. Confirmed words go into the known list and become anchors in the next round; rejected words are added to a known-normal list.

This keeps the manual work small, and every review round makes the next one better (more anchors, fewer repeated false alarms).

## 6. Full pipeline

**Once, now:**

1. Train fastText on the Jigsaw training comments (settings in 3.1).
2. Build the anchors (3.2).
3. Run the leave-out test and fix the threshold (3.5); 0.58 is the safer value from our run.

**Each time new text arrives:**

4. Load the model and continue training on **only the new text**:
   ```python
   from gensim.models import FastText
   model = FastText.load("slang/fasttext.model")
   model.build_vocab(new_sentences, update=True)   # add the new words
   model.train(new_sentences, total_examples=len(new_sentences), epochs=5)
   ```
5. Candidates = words that appear in the new text at least 5 times and are not on the known list.
6. Score them (3.3) and keep those with score ≥ threshold.
7. Human review of the top of the ranking (section 5); confirmed words become anchors.
8. Collect comments containing confirmed words, label them, and fine-tune the classifier (section 2).

Run the proof of concept:

```bash
python slang_fasttext.py --data out --profanity ldnoobw_en.txt --out slang
python slang_fasttext.py --data out --profanity ldnoobw_en.txt --out slang_w2v --kind word2vec   # comparison
```

## 7. Limitations

- **No time information.** The Jigsaw comments have no reliable dates (timestamps are replaced by `timetoken`), so we cannot show slang *emerging over time*, only words that behave like known cuss words. The "new text" run uses the test comments as a stand-in.
- **The leave-out test is easier than real use.** Hidden anchors like `fucker` are frequent and often spelled like visible anchors (`fuck`). Real new slang starts rare and may look like nothing known, so expect lower real-world recall.
- **Rare words are invisible.** A new word needs at least `min_count` occurrences before it can be scored, so detection lags behind the first uses.
- **One vector per word.** fastText averages all uses of a word. If a word is still mostly used in its old meaning, the shift toward insult usage looks weaker.
- **Drift.** Training only on new text slowly moves old vectors too, including the anchors. Use few epochs for updates and check that anchors stay similar to each other after each update.
- **Numbers are lost in the TF-IDF branch.** A number used as an insult (e.g. `403`) becomes `numtoken` there; the monitor itself keeps numbers, but the classifier would need the BERT branch or labeled data to use it.

## 8. Possible next steps

- Insult-focused anchors (from our own filtered $r$ list) instead of the mostly sexual public list, to reduce the sexuality-word bias.
- Start from a full pretrained fastText model (`.bin`) and continue training on Jigsaw, for broader vocabulary.
- Collect real time-stamped text (e.g. recent forum comments) to test detection of genuinely new slang.
- Use BERT's contextual vectors (averaged per word) as an alternative that shares the classifier's embedding space.
