"""
Cross-validation experiment: DeepSparseMLP (DeepMLPModel) with the BagOfWords encoder
on LabHC/bias_in_bios (profession classification).

Setup follows project.ipynb: BPE tokenizer (vocab_size=1000, min_frequency=2) trained on the
cleaned train split, BagOfWords encoder, 5-fold CV on the train split (cross_validation_split
from the tutorial), accuracy (+ macro-F1) per fold.

Usage: python run_deep_bow.py [--num_epochs N] [--subset N]
"""
import argparse
import itertools
import json
import random
import time

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

import encoders
import models
import utils


def cross_validation_split(pairs, k, seed=0):
    """Same as the tutorial: shuffle and split into k folds."""
    random.seed(seed)
    pairs = list(pairs)
    ids = list(range(len(pairs)))
    random.shuffle(ids)
    ids_parts = np.array_split(np.array(ids, dtype=object), k)
    cv = {}
    for i in range(k):
        training_ids = list(itertools.chain.from_iterable(p for j, p in enumerate(ids_parts) if j != i))
        cv[i] = {
            "training": [pairs[j] for j in training_ids],
            "validation": [pairs[j] for j in ids_parts[i]],
        }
    return cv


def fmt(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}u{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class Progress:
    """Tracks the whole run and prints what is happening plus an ETA for the remaining training."""

    def __init__(self, num_folds, num_epochs):
        self.num_folds, self.num_epochs = num_folds, num_epochs
        self.start = time.time()
        self.fold = 0
        self.epoch = 0
        self.fold_start = None
        self.fold_times = []    # total wall time per finished fold (incl. encoding/prediction)
        self.epoch_times = []   # wall time per epoch (train + validation)
        self._epoch_start = None

    def phase(self, msg):
        print(f"[{fmt(time.time() - self.start)}] {msg}", flush=True)

    def start_fold(self, fold):
        self.fold, self.epoch = fold, 0
        self.fold_start = time.time()
        self.phase(f"Fold {fold + 1}/{self.num_folds}: documenten encoderen + model trainen")

    def end_fold(self):
        self.fold_times.append(time.time() - self.fold_start)

    def epoch_started(self):
        self._epoch_start = time.time()

    def epoch_finished(self):
        self.epoch_times.append(time.time() - self._epoch_start)
        self.epoch += 1
        avg_epoch = np.mean(self.epoch_times[-self.num_epochs:])
        rest_this_fold = (self.num_epochs - self.epoch) * avg_epoch
        folds_left = self.num_folds - self.fold - 1
        if self.fold_times:
            rest_other = folds_left * np.mean(self.fold_times)
        else:  # no full fold finished yet: estimate a fold from epoch time + overhead so far
            overhead = (self._epoch_start - self.fold_start) - sum(self.epoch_times[:-1])
            rest_other = folds_left * (self.num_epochs * avg_epoch + max(overhead, 0))
        eta = rest_this_fold + rest_other
        print(f"    -> fold {self.fold + 1}/{self.num_folds}, epoch {self.epoch}/{self.num_epochs} klaar "
              f"({avg_epoch:.1f}s/epoch) | verstreken {fmt(time.time() - self.start)} | "
              f"geschat resterend {fmt(eta)} | klaar rond {time.strftime('%H:%M', time.localtime(time.time() + eta))}",
              flush=True)


def install_progress_hooks(progress):
    """Wrap models.train_one_epoch / models.evaluate so every epoch reports progress (models.py stays untouched)."""
    train_one_epoch, evaluate = models.train_one_epoch, models.evaluate

    def timed_train(*a, **kw):
        progress.epoch_started()
        return train_one_epoch(*a, **kw)

    def timed_eval(*a, **kw):
        out = evaluate(*a, **kw)
        progress.epoch_finished()
        return out

    models.train_one_epoch, models.evaluate = timed_train, timed_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_epochs", type=int, default=20)
    ap.add_argument("--num_folds", type=int, default=5)
    ap.add_argument("--subset", type=int, default=None, help="use only the first N training examples (quick test)")
    ap.add_argument("--out", default="results_deep_bow.json")
    args = ap.parse_args()

    progress = Progress(args.num_folds, args.num_epochs)
    install_progress_hooks(progress)

    progress.phase("Dataset laden en opschonen...")
    ds = load_dataset("LabHC/bias_in_bios")
    texts = utils.clean_corpus(ds["train"]["hard_text"])
    labels = list(ds["train"]["profession"])
    if args.subset:
        texts, labels = texts[:args.subset], labels[:args.subset]
    num_classes = 28

    # Tokenizer exactly as in project.ipynb
    progress.phase(f"BPE tokenizer trainen op {len(texts)} teksten...")
    tokenizer = encoders.train_tokenizer(texts, vocab_size=1000, min_frequency=2)
    encoder = encoders.BagOfWords(tokenizer)
    print(f"Vocab size: {len(encoder)}")

    cv = cross_validation_split(list(zip(texts, labels)), k=args.num_folds, seed=0)

    results = {"config": {**vars(args), "model": "DeepMLPModel", "hidden_dims": models.DeepMLPModel.default_hidden_dims,
                          "encoder": "BagOfWords", "vocab_size": 1000, "min_frequency": 2}, "folds": []}
    for fold, split in cv.items():
        print(f"\n===== Fold {fold + 1}/{args.num_folds} =====")
        t0 = time.time()
        progress.start_fold(fold)
        model = models.DeepMLPModel.fit(split["training"], encoder, num_classes=num_classes,
                                        num_epochs=args.num_epochs, seed=42)
        progress.phase(f"Fold {fold + 1}: voorspellen op validatiefold...")
        val_texts = [t for t, _ in split["validation"]]
        y_true = [y for _, y in split["validation"]]
        y_pred = models.DeepMLPModel.predict(val_texts, model, encoder)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        progress.end_fold()
        print(f"Fold {fold + 1}: accuracy={acc:.4f} macro-F1={f1:.4f} best_epoch={model.best_epoch + 1} "
              f"({time.time() - t0:.0f}s)")
        results["folds"].append({"fold": fold, "accuracy": acc, "macro_f1": f1,
                                 "best_epoch": model.best_epoch, "history": model.history})
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    accs = np.array([r["accuracy"] for r in results["folds"]])
    f1s = np.array([r["macro_f1"] for r in results["folds"]])
    print("\n===== Summary DeepSparseMLP + BagOfWords =====")
    for r in results["folds"]:
        print(f"fold {r['fold'] + 1}: acc {r['accuracy']:.4f} | macro-F1 {r['macro_f1']:.4f} | best epoch {r['best_epoch'] + 1}")
    print(f"accuracy: {accs.mean():.4f} ± {accs.std(ddof=1):.4f}")
    print(f"macro-F1: {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f}")
    progress.phase("Klaar.")


if __name__ == "__main__":
    main()
