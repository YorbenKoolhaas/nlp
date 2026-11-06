"""
5-fold cross-validation on the TRAIN split of LabHC/bias_in_bios for all
3 encoders (indicator, bow, tfidf) x 3 models (loglin, mlp, deep).

    python run_cv.py              # runs every combination that is not in results/ yet
    python run_cv.py --overwrite  # redo everything

Output: results/cv_{encoder}_{model}.json with accuracy and macro-F1 per fold.
"""
import argparse
import itertools
import json
import os
import random
import time
import warnings

import numpy as np
from datasets import load_dataset
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score

import encoders
import models
import utils

ENCODERS = ["indicator", "bow", "tfidf"]
MODELS = ["loglin", "mlp", "deep"]
VOCAB_SIZE, MIN_FREQUENCY = 8000, 2
NUM_FOLDS, NUM_EPOCHS, NUM_CLASSES = 5, 5, 28
RESULTS_DIR = "results"


def load_split(name):
    ds = load_dataset("LabHC/bias_in_bios")[name]
    return utils.clean_corpus(ds["hard_text"]), list(ds["profession"])


def cross_validation_split(pairs, k, seed=0):
    """From the tutorial: shuffle and split into k folds."""
    random.seed(seed)
    ids = list(range(len(pairs)))
    random.shuffle(ids)
    parts = np.array_split(np.array(ids), k)
    return [{"training": [pairs[j] for j in itertools.chain(*(p for n, p in enumerate(parts) if n != i))],
             "validation": [pairs[j] for j in parts[i]]} for i in range(k)]


def count_matrix(tokenizer, texts):
    """Tokenise every text once: returns {text: row} and a sparse [num_texts, vocab] matrix of token counts."""
    ids = [e.ids for e in tokenizer.encode_batch(texts)]
    rows = np.repeat(np.arange(len(ids)), [len(x) for x in ids])
    counts = csr_matrix((np.ones(sum(map(len, ids))), (rows, np.concatenate(ids))),
                        shape=(len(texts), tokenizer.get_vocab_size()))
    counts.sum_duplicates()
    return {t: i for i, t in enumerate(texts)}, counts


class Features:
    """Encoder with the same interface as encoders.py (len(enc), enc(text)), looking rows up in a precomputed
    sparse matrix. Sparse, because dense vectors for vocab 8000 would not fit in memory."""

    def __init__(self, index, matrix):
        self.index, self.matrix = index, matrix

    def __len__(self):
        return self.matrix.shape[1]

    def __call__(self, text):
        return self.matrix[self.index[text]]


def make_encoder(name, index, counts, training_texts):
    """Same features as encoders.TokenIndicator / BagOfWords / TfIdfEncoder."""
    if name == "bow":
        return Features(index, counts)
    if name == "indicator":
        m = counts.copy()
        m.data[:] = 1.0
        return Features(index, m)
    # tfidf: tf = count / doc length, idf from the training texts only, then L2-normalised per document
    df = np.asarray((counts[[index[t] for t in training_texts]] > 0).sum(axis=0)).ravel()
    idf = np.log(len(training_texts) / np.maximum(df, 1))
    x = csr_matrix(csr_matrix(counts.multiply(1.0 / counts.sum(axis=1))).multiply(idf))
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1))).ravel()
    return Features(index, csr_matrix(x.multiply(1.0 / np.maximum(norms, 1e-12)[:, None])))


def fit_predict(model_name, training_set, test_texts, encoder, validation_set=None):
    """Train one model and predict test_texts. MLPs pick their best epoch on validation_set
    (if None: on 10% of training_set)."""
    if model_name == "loglin":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = models.LogLinearModel.fit(training_set, encoder)
        return models.LogLinearModel.predict(test_texts, model, encoder)
    wrapper = models.DeepMLPModel if model_name == "deep" else models.MLPModel
    model = wrapper.fit(training_set, encoder, validation_set=validation_set,
                        num_classes=NUM_CLASSES, num_epochs=NUM_EPOCHS)
    return wrapper.predict(test_texts, model, encoder)


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m // 60}u{m % 60:02d}m" if m >= 60 else f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="redo combinations that already have results")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = lambda e, m: os.path.join(RESULTS_DIR, f"cv_{e}_{m}.json")
    todo = [(e, m) for m in MODELS for e in ENCODERS if args.overwrite or not os.path.exists(path(e, m))]
    if not todo:
        return print("Alles staat al in results/ (gebruik --overwrite om opnieuw te draaien).")

    start = time.time()
    print("Data laden, tokenizer trainen en alles tokeniseren...")
    texts, labels = load_split("train")
    tokenizer = encoders.train_tokenizer(texts, vocab_size=VOCAB_SIZE, min_frequency=MIN_FREQUENCY)
    index, counts = count_matrix(tokenizer, texts)
    folds = cross_validation_split(list(zip(texts, labels)), NUM_FOLDS)

    fold_times, total = [], len(todo) * NUM_FOLDS
    for r, (enc_name, model_name) in enumerate(todo):
        scores = []
        for k, fold in enumerate(folds):
            print(f"\n[{fmt(time.time() - start)}] run {r + 1}/{len(todo)} ({enc_name} + {model_name}), fold {k + 1}/{NUM_FOLDS}...")
            t0 = time.time()
            encoder = make_encoder(enc_name, index, counts, [t for t, _ in fold["training"]])
            y_true = [y for _, y in fold["validation"]]
            y_pred = fit_predict(model_name, fold["training"], [t for t, _ in fold["validation"]], encoder)
            scores.append({"accuracy": accuracy_score(y_true, y_pred),
                           "macro_f1": f1_score(y_true, y_pred, average="macro")})

            fold_times.append(time.time() - t0)
            left = np.mean(fold_times) * (total - len(fold_times))
            print(f"[{fmt(time.time() - start)}] accuracy {scores[-1]['accuracy']:.4f}, macro-F1 {scores[-1]['macro_f1']:.4f}"
                  f" | nog ~{fmt(left)}, klaar rond {time.strftime('%H:%M', time.localtime(time.time() + left))}")

        with open(path(enc_name, model_name), "w") as f:
            json.dump({"encoder": enc_name, "model": model_name, "folds": scores}, f, indent=2)
        acc = [s["accuracy"] for s in scores]
        print(f">>> {enc_name} + {model_name}: accuracy {np.mean(acc):.4f} ± {np.std(acc, ddof=1):.4f}")

    print(f"\nKlaar in {fmt(time.time() - start)}. Volgende stap: python final_test.py")


if __name__ == "__main__":
    main()
