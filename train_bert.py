"""
Fine-tune an uncased BERT as a 6-label classifier (one sigmoid per label, binary cross-entropy).

Needs the outputs of preprocess.py (out/train_clean.csv, out/test_clean.csv, out/folds.csv).
Use a GPU (Colab T4 or better). On CPU this only makes sense for a smoke test.

Usage (Colab):
  pip install transformers scikit-learn pandas
  python train_bert.py --data out --preds preds                        # full run
  python train_bert.py --data out --preds preds --subset 20000         # quick run on 20k comments
  python train_bert.py --model distilbert-base-uncased --max-len 192   # faster model
  python train_bert.py --data out --preds preds --pos-weight --tag bert_pw   # rare labels weighted more

Writes to preds/ (file names start with --tag, default "bert"):
  bert_val.csv    predictions on the validation fold (best epoch)
  bert_test.csv   predictions on the labeled test set (best epoch)
  bert_best/      the best model + tokenizer

--pos-weight: binary cross-entropy where a positive example of label j counts w_j times,
w_j = (#negatives / #positives) of that label in the training split, capped at --pos-weight-cap.
Rare labels (threat ~0.3%) then push the model as hard as common ones during training.
Pick cut-offs AFTER training (evaluate.py --val-pred ...), since the weighted model's
probabilities are shifted upwards.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from evaluate import auc_table
from preprocess import LABELS


def make_loader(texts, labels, tok, max_len, batch, shuffle):
    """Tokenize per batch with dynamic padding (pad to the longest comment in the batch, not to max_len)."""
    idx = np.arange(len(texts))

    def collate(b):
        enc = tok([texts[i] for i in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor(labels[b], dtype=torch.float)
        return enc

    if not shuffle:   # sort by length for faster inference, keep track of the original order
        idx = idx[np.argsort([len(t) for t in texts])]
    return DataLoader(idx, batch_size=batch, shuffle=shuffle, collate_fn=lambda b: collate(np.array(b))), idx


@torch.no_grad()
def predict(model, texts, tok, max_len, batch, device, amp):
    model.eval()
    loader, order = make_loader(texts, None, tok, max_len, batch * 2, shuffle=False)
    out = []
    for enc in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device.type, enabled=amp):
            out.append(torch.sigmoid(model(**enc).logits.float()).cpu().numpy())
    probs = np.empty((len(texts), len(LABELS)), dtype=np.float32)
    probs[order] = np.concatenate(out)
    return probs


def to_frame(ids, probs):
    return pd.concat([pd.DataFrame({"id": ids}), pd.DataFrame(probs, columns=LABELS)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out")
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help="train on a random subset of N comments (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos-weight", action="store_true", help="weight positives per label (see top of file)")
    ap.add_argument("--pos-weight-cap", type=float, default=50.0)
    ap.add_argument("--tag", default="bert", help="prefix for output files, e.g. bert_pw")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"                    # mixed precision on GPU: ~2x faster
    preds = Path(a.preds)
    preds.mkdir(parents=True, exist_ok=True)
    print("device:", device)

    # ---- data
    data = Path(a.data)
    train = pd.read_csv(data / "train_clean.csv").merge(pd.read_csv(data / "folds.csv"), on="id")
    test = pd.read_csv(data / "test_clean.csv")
    for df in (train, test):
        df["text_bert"] = df.text_bert.fillna("")
    tr, va = train[train.fold != a.val_fold], train[train.fold == a.val_fold].reset_index(drop=True)
    if a.subset:
        tr = tr.sample(min(a.subset, len(tr)), random_state=a.seed)
        va = va.sample(min(a.subset // 4, len(va)), random_state=a.seed).reset_index(drop=True)
    print(f"train {len(tr):,}  val {len(va):,}  test {len(test):,}")

    # ---- model
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(LABELS), problem_type="multi_label_classification").to(device)

    loader, _ = make_loader(tr.text_bert.tolist(), tr[LABELS].values, tok, a.max_len, a.batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(loader) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    pos_weight = None
    if a.pos_weight:
        share = tr[LABELS].values.mean(axis=0)
        pos_weight = torch.tensor(np.clip((1 - share) / share, 1.0, a.pos_weight_cap),
                                  dtype=torch.float, device=device)
        print("pos_weight:", {c: round(float(w), 1) for c, w in zip(LABELS, pos_weight.cpu())})
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler(enabled=amp)

    # ---- train
    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for i, enc in enumerate(loader, 1):
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = enc.pop("labels")
            with torch.autocast(device.type, enabled=amp):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), labels)    # (weighted) BCE over the 6 labels
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            if i % 200 == 0 or i == len(loader):
                print(f"ep {ep} step {i}/{len(loader)} loss {run_loss / i:.4f} ({time.time() - t0:.0f}s)")

        # ---- validate, keep the best epoch
        p_va = predict(model, va.text_bert.tolist(), tok, a.max_len, a.batch, device, amp)
        table = auc_table(va[LABELS], pd.DataFrame(p_va, columns=LABELS), va.comment_text)
        print(table.round(4).to_string())
        score = table.loc["all", "mean"]
        if score > best:
            best = score
            model.save_pretrained(preds / f"{a.tag}_best")
            tok.save_pretrained(preds / f"{a.tag}_best")
            to_frame(va.id, p_va).to_csv(preds / f"{a.tag}_val.csv", index=False)
            print(f"ep {ep}: new best val mean AUC {best:.4f}, saved")

    # ---- test with the best model
    model = AutoModelForSequenceClassification.from_pretrained(preds / f"{a.tag}_best").to(device)
    p_te = predict(model, test.text_bert.tolist(), tok, a.max_len, a.batch, device, amp)
    to_frame(test.id, p_te).to_csv(preds / f"{a.tag}_test.csv", index=False)
    print("wrote", preds / f"{a.tag}_test.csv")


if __name__ == "__main__":
    main()
