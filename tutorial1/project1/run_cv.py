"""
k-fold cross-validation for every (encoder, model) combination on LabHC/bias_in_bios (profession).

Setup: BPE tokenizer (default vocab_size=8000, min_frequency=2) trained on the cleaned train split
(project.ipynb uses 1000; pass --vocab_size 1000 to reproduce that); folds from the tutorial's cross_validation_split (seed 0), so every system is evaluated on
exactly the same folds (needed for the paired Bayesian comparison in analyse_cv.py).

Encoders: indicator (TokenIndicator), bow (BagOfWords), tfidf (TfIdfEncoder, idf fitted on the training fold)
Models:   loglin (LogLinearModel), mlp (MLPModel, SparseMLP), deep (DeepMLPModel, DeepSparseMLP [256, 128])

Examples:
    python run_cv.py --encoder bow --model deep      # one combination
    python run_cv.py --all                           # all 9 combinations (already finished ones are skipped)
    python run_cv.py --all --overwrite               # redo everything

Results: results/vocab{V}[_lower]/cv_{encoder}_{model}.json (per-fold accuracy/macro-F1, plus training history for the MLPs).
"""
import argparse
import itertools
import json
import os
import random
import threading
import time
import warnings

import numpy as np
from datasets import load_dataset
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, f1_score

import encoders
import models
import utils

ENCODERS = ["indicator", "bow", "tfidf"]
MODELS = ["loglin", "mlp", "deep"]
NUM_CLASSES = 28


# ----------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------
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


class TokenCounts:
    """
    Tokenises every text ONCE (tokenizer.encode_batch, parallel) into a sparse [num_docs, vocab] count matrix.
    All three encoders are functions of these counts, so they no longer re-tokenise every document for every
    fold and every model (that alone would take hours at full scale).
    """

    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.index = {}                       # text -> row (identical texts have identical counts)
        for i, t in enumerate(texts):
            self.index.setdefault(t, i)
        self.counts = self._count_matrix([e.ids for e in tokenizer.encode_batch(texts)])
        self.lengths = np.asarray(self.counts.sum(axis=1)).ravel()   # number of tokens per document

    def _count_matrix(self, id_lists):
        rows = np.repeat(np.arange(len(id_lists)), [len(ids) for ids in id_lists])
        cols = np.fromiter(itertools.chain.from_iterable(id_lists), dtype=np.int64)
        m = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(id_lists), self.vocab_size))
        m.sum_duplicates()
        return m

    def rows(self, texts):
        return np.array([self.index[t] for t in texts])


class CachedEncoder:
    """
    Encoder (same interface as encoders.py: len() and encoder(text)) that looks a document up in a
    precomputed sparse feature matrix and returns a 1 x D sparse row. Texts that were not precomputed
    are tokenised on the fly.
    """

    def __init__(self, counts: TokenCounts, transform):
        self.c, self.transform = counts, transform
        self.features = transform(counts.counts, counts.lengths)

    def __len__(self):
        return self.c.vocab_size

    def __call__(self, text):
        i = self.c.index.get(text)
        if i is not None:
            return self.features[i]
        row = self.c._count_matrix([self.c.tokenizer.encode(text).ids])
        return self.transform(row, np.asarray(row.sum(axis=1)).ravel())


def indicator_transform(counts, lengths):
    """encoders.TokenIndicator: 1 if the token occurs in the document."""
    m = counts.copy()
    m.data[:] = 1.0
    return m


def bow_transform(counts, lengths):
    """encoders.BagOfWords: raw token counts."""
    return counts


def tfidf_transform_factory(counts: TokenCounts, training_rows):
    """
    encoders.TfIdfEncoder: tf = count / len(doc), idf = log(N / df), with df and N from the training fold only,
    then every document vector is L2-normalised.
    Tokens that never occur in the training fold get idf = log(N) (as if df = 1).
    """
    train = counts.counts[training_rows]
    df = np.asarray((train > 0).sum(axis=0)).ravel()
    n = len(training_rows)
    idf = np.where(df > 0, np.log(n / np.maximum(df, 1)), np.log(n))

    def transform(c, lengths):
        tf = csr_matrix(c.multiply(1.0 / lengths[:, None]))
        x = csr_matrix(tf.multiply(idf[None, :]))
        # L2 normalisation per document, same as encoders.TfIdfEncoder
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        return csr_matrix(x.multiply(1.0 / np.where(norms > 0, norms, 1.0)[:, None]))
    return transform


def make_encoder(name, counts: TokenCounts, training_set):
    if name == "indicator":
        return CachedEncoder(counts, indicator_transform)
    if name == "bow":
        return CachedEncoder(counts, bow_transform)
    if name == "tfidf":
        # idf only from the training fold, so the validation fold does not leak into the features
        return CachedEncoder(counts, tfidf_transform_factory(counts, counts.rows([t for t, _ in training_set])))
    raise ValueError(name)


def verify_against_original_encoders(counts: TokenCounts, texts, n_check=200, n_idf=2000):
    """Check on a sample that the cached features are identical to the encoders in encoders.py."""
    rng = np.random.default_rng(0)
    sample = [texts[i] for i in rng.choice(len(texts), size=min(n_check, len(texts)), replace=False)]
    idf_corpus = texts[:n_idf]
    original = {
        "indicator": encoders.TokenIndicator(counts.tokenizer),
        "bow": encoders.BagOfWords(counts.tokenizer),
        "tfidf": encoders.TfIdfEncoder(counts.tokenizer, idf_corpus),
    }
    for token_id in counts.tokenizer.get_vocab().values():
        original["tfidf"].idf.setdefault(token_id, np.log(len(idf_corpus)))
    cached = {
        "indicator": make_encoder("indicator", counts, None),
        "bow": make_encoder("bow", counts, None),
        "tfidf": make_encoder("tfidf", counts, [(t, None) for t in idf_corpus]),
    }
    for name in original:
        for t in sample:
            a, b = original[name](t), cached[name](t).toarray().ravel()
            if not np.allclose(a, b):
                raise AssertionError(f"cached {name} features differ from encoders.py for text: {t[:80]!r}")
    print(f"Controle: cached features identiek aan encoders.py ({len(sample)} documenten x 3 encoders)", flush=True)


# ----------------------------------------------------------------------------------------------------------
# Progress / ETA
# ----------------------------------------------------------------------------------------------------------
def fmt(seconds):
    if seconds is None:
        return "onbekend"
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}u{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class Progress:
    """
    Keeps track of all planned runs and prints what is happening plus the expected remaining time for
    the current run and for everything that is still planned. Prints after every epoch (MLPs), after every
    fold, and every 30 s as a heartbeat (useful for logistic regression, which has no epochs).
    """

    def __init__(self, runs, num_folds, num_epochs):
        self.runs = runs                  # list of (encoder, model) still to do
        self.num_folds, self.num_epochs = num_folds, num_epochs
        self.start = time.time()
        self.run_idx = 0
        self.fold = 0
        self.epoch = 0
        self.fold_start = None
        self.fold_times = {m: [] for m in MODELS}   # wall time per finished fold, per model type
        self.epoch_times = []
        self._epoch_start = None
        self.status = "opstarten"
        self.last_print = time.time()
        self._lock = threading.Lock()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    # --- estimates ------------------------------------------------------------------------------------
    def _est_fold(self, model):
        if self.fold_times[model]:
            return np.mean(self.fold_times[model])
        if model != "loglin" and self.epoch_times and self.current_model != "loglin":
            # estimate an MLP fold from the current epoch speed (+ overhead seen in the current fold)
            return self.num_epochs * np.mean(self.epoch_times[-self.num_epochs:]) + 60
        known = [t for ts in self.fold_times.values() for t in ts]
        return np.mean(known) if known else None

    @property
    def current_model(self):
        return self.runs[self.run_idx][1]

    def eta(self):
        """(remaining for current run, remaining for everything)"""
        model = self.current_model
        in_fold = time.time() - self.fold_start if self.fold_start else 0
        if model != "loglin" and self.epoch_times and self.epoch > 0:
            rest_fold = (self.num_epochs - self.epoch) * np.mean(self.epoch_times[-self.num_epochs:]) + 10
        else:
            est = self._est_fold(model)
            rest_fold = None if est is None else max(est - in_fold, 0)
        est = self._est_fold(model)
        if rest_fold is None or est is None:
            return None, None
        rest_run = rest_fold + (self.num_folds - self.fold - 1) * est
        rest_all = rest_run
        for _, m in self.runs[self.run_idx + 1:]:
            e = self._est_fold(m)
            if e is None:
                return rest_run, None
            rest_all += self.num_folds * e
        return rest_run, rest_all

    # --- printing -------------------------------------------------------------------------------------
    def line(self, msg=None):
        with self._lock:
            enc, model = self.runs[self.run_idx]
            rest_run, rest_all = self.eta()
            done_at = "" if rest_all is None else \
                f" | alles klaar rond {time.strftime('%H:%M', time.localtime(time.time() + rest_all))}"
            print(f"[{fmt(time.time() - self.start)}] run {self.run_idx + 1}/{len(self.runs)} ({enc} + {model}) "
                  f"fold {self.fold + 1}/{self.num_folds} | {msg or self.status} | "
                  f"rest deze run: {fmt(rest_run)} | rest totaal: {fmt(rest_all)}{done_at}", flush=True)
            self.last_print = time.time()

    def phase(self, msg):
        print(f"[{fmt(time.time() - self.start)}] {msg}", flush=True)
        self.last_print = time.time()

    def _heartbeat(self):
        while True:
            time.sleep(5)
            if self.fold_start is not None and time.time() - self.last_print >= 30:
                self.line()

    # --- events ---------------------------------------------------------------------------------------
    def start_run(self, idx):
        self.run_idx, self.fold, self.epoch = idx, 0, 0
        enc, model = self.runs[idx]
        print(f"\n{'=' * 100}\nRun {idx + 1}/{len(self.runs)}: encoder={enc}, model={model}\n{'=' * 100}", flush=True)

    def start_fold(self, fold):
        self.fold, self.epoch = fold, 0
        self.fold_start = time.time()
        self.status = "documenten encoderen + trainen" if self.current_model != "loglin" \
            else "documenten encoderen + logistische regressie trainen (geen epochs, heartbeat elke 30s)"
        self.line()

    def end_fold(self, acc, f1):
        self.fold_times[self.current_model].append(time.time() - self.fold_start)
        self.line(f"fold klaar: accuracy {acc:.4f}, macro-F1 {f1:.4f} ({fmt(self.fold_times[self.current_model][-1])})")

    def epoch_started(self):
        self._epoch_start = time.time()

    def epoch_finished(self, val_metrics):
        self.epoch_times.append(time.time() - self._epoch_start)
        self.epoch += 1
        self.status = f"epoch {self.epoch}/{self.num_epochs} bezig"
        self.line(f"epoch {self.epoch}/{self.num_epochs} klaar ({self.epoch_times[-1]:.1f}s, "
                  f"val loss {val_metrics['loss']:.4f}, val acc {val_metrics['accuracy']:.4f})")


def install_progress_hooks(progress):
    """Wrap models.train_one_epoch / models.evaluate so every epoch reports progress (models.py stays untouched)."""
    train_one_epoch, evaluate = models.train_one_epoch, models.evaluate

    def timed_train(*a, **kw):
        progress.epoch_started()
        return train_one_epoch(*a, **kw)

    def timed_eval(*a, **kw):
        metrics, preds = evaluate(*a, **kw)
        progress.epoch_finished(metrics)
        return metrics, preds

    models.train_one_epoch, models.evaluate = timed_train, timed_eval


# ----------------------------------------------------------------------------------------------------------
# One fold
# ----------------------------------------------------------------------------------------------------------
def fit_and_predict(model_name, training_set, val_texts, encoder, num_epochs, progress, max_iter=100):
    """Returns predicted labels for val_texts and extra info (history / best epoch for the MLPs)."""
    if model_name == "loglin":
        progress.status = "documenten encoderen + logistische regressie trainen"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # saga with max_iter=100, as in the tutorial
            model = models.LogLinearModel.fit(training_set, encoder, max_iter=max_iter)
        progress.status = "voorspellen op validatiefold"
        preds = models.LogLinearModel.predict(val_texts, model, encoder)  # returns labels
        n_iter = int(np.max(model.n_iter_))
        converged = n_iter < max_iter
        progress.phase(f"    logistische regressie: {n_iter}/{max_iter} iteraties, "
                       f"{'geconvergeerd' if converged else 'NIET geconvergeerd (verhoog --max_iter)'}")
        return preds, {"n_iter": n_iter, "converged": converged}

    wrapper = models.DeepMLPModel if model_name == "deep" else models.MLPModel
    model = wrapper.fit(training_set, encoder, num_classes=NUM_CLASSES, num_epochs=num_epochs, seed=42)
    progress.status = "voorspellen op validatiefold"
    progress.line()
    preds = wrapper.predict(val_texts, model, encoder)
    return preds, {"best_epoch": model.best_epoch, "history": model.history}


# ----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=ENCODERS)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--all", action="store_true", help="run all encoder x model combinations")
    ap.add_argument("--num_epochs", type=int, default=5, help="epochs for the MLPs (best epoch was 2-3 with 20)")
    ap.add_argument("--max_iter", type=int, default=100, help="max saga iterations for loglin (tutorial default: 100)")
    ap.add_argument("--num_folds", type=int, default=5)
    ap.add_argument("--subset", type=int, default=None, help="only the first N training examples (quick test)")
    ap.add_argument("--vocab_size", type=int, default=8000, help="BPE vocabulary size (project.ipynb used 1000)")
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--lowercase", action="store_true", help="lowercase all texts before tokenisation")
    ap.add_argument("--results_dir", default=None, help="default: results/vocab{V}[_lower]")
    ap.add_argument("--overwrite", action="store_true", help="redo runs whose result file already exists")
    args = ap.parse_args()

    if args.all:
        combos = [(e, m) for m in MODELS for e in ENCODERS]
    elif args.encoder and args.model:
        combos = [(args.encoder, args.model)]
    else:
        ap.error("give --encoder and --model, or --all")

    if args.results_dir is None:
        args.results_dir = os.path.join("results", f"vocab{args.vocab_size}" + ("_lower" if args.lowercase else ""))
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Resultaten naar: {args.results_dir}/")
    path = lambda e, m: os.path.join(args.results_dir, f"cv_{e}_{m}.json")
    todo = [(e, m) for e, m in combos if args.overwrite or not os.path.exists(path(e, m))]
    for e, m in combos:
        if (e, m) not in todo:
            print(f"Overslaan (al klaar): {path(e, m)}  -- gebruik --overwrite om opnieuw te draaien")
    if not todo:
        print("Niets te doen.")
        return
    print("Te draaien: " + ", ".join(f"{e}+{m}" for e, m in todo))

    progress = Progress(todo, args.num_folds, args.num_epochs)
    install_progress_hooks(progress)

    progress.phase("Dataset laden en opschonen...")
    ds = load_dataset("LabHC/bias_in_bios")
    texts = utils.clean_corpus(ds["train"]["hard_text"])
    labels = list(ds["train"]["profession"])
    if args.lowercase:
        texts = [t.lower() for t in texts]
    if args.subset:
        texts, labels = texts[:args.subset], labels[:args.subset]

    progress.phase(f"BPE tokenizer trainen op {len(texts)} teksten (vocab {args.vocab_size}, "
                   f"min_frequency {args.min_frequency}{', lowercase' if args.lowercase else ''})...")
    tokenizer = encoders.train_tokenizer(texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency)

    progress.phase("Alle teksten één keer tokeniseren naar een sparse tellingenmatrix...")
    counts = TokenCounts(tokenizer, texts)
    print(f"    {counts.counts.shape[0]} documenten x {counts.vocab_size} tokens, "
          f"gem. {counts.lengths.mean():.0f} tokens/doc, {counts.counts.nnz} niet-nul", flush=True)
    verify_against_original_encoders(counts, texts)

    progress.phase("Folds maken...")
    cv = cross_validation_split(list(zip(texts, labels)), k=args.num_folds, seed=0)

    for run_idx, (enc_name, model_name) in enumerate(todo):
        progress.start_run(run_idx)
        result = {
            "config": {"encoder": enc_name, "model": model_name, "num_folds": args.num_folds,
                       "num_epochs": args.num_epochs if model_name != "loglin" else None,
                       "max_iter": args.max_iter if model_name == "loglin" else None,
                       "hidden_dims": models.DeepMLPModel.default_hidden_dims if model_name == "deep" else None,
                       "vocab_size": args.vocab_size, "min_frequency": args.min_frequency,
                       "lowercase": args.lowercase, "cv_seed": 0, "subset": args.subset},
            "folds": [],
        }
        for fold, split in cv.items():
            progress.start_fold(fold)
            if enc_name == "tfidf":
                progress.status = "idf berekenen op trainingsfold"
            encoder = make_encoder(enc_name, counts, split["training"])
            val_texts = [t for t, _ in split["validation"]]
            y_true = [y for _, y in split["validation"]]

            t0 = time.time()
            y_pred, extra = fit_and_predict(model_name, split["training"], val_texts, encoder,
                                            args.num_epochs, progress, max_iter=args.max_iter)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            result["folds"].append({"fold": fold, "accuracy": acc, "macro_f1": f1,
                                    "seconds": time.time() - t0, **extra})
            progress.end_fold(acc, f1)

        with open(path(enc_name, model_name), "w") as f:
            json.dump(result, f, indent=2)
        accs = [r["accuracy"] for r in result["folds"]]
        f1s = [r["macro_f1"] for r in result["folds"]]
        print(f"\n>>> {enc_name} + {model_name}: accuracy {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f} | "
              f"macro-F1 {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}  -> {path(enc_name, model_name)}", flush=True)

    progress.phase(f"Alles klaar. Analyse: python analyse_cv.py --results_dir {args.results_dir}")


if __name__ == "__main__":
    main()
