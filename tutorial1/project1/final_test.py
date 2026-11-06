"""
Final evaluation on the held-out TEST split for all 9 (encoder, model) combinations:
train on the full train split, MLPs pick their best epoch on dev, evaluate once on test.

    python final_test.py

Output: results/final_test.json, plus a table with the test scores next to the CV scores.
"""
import json
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

import encoders
from run_cv import (ENCODERS, MIN_FREQUENCY, MODELS, RESULTS_DIR, VOCAB_SIZE, count_matrix, fit_predict, fmt,
                    load_split, make_encoder)


def main():
    start = time.time()
    print("Data laden, tokenizer trainen (alleen op train) en alles tokeniseren...")
    (train_x, train_y), (dev_x, dev_y), (test_x, test_y) = (load_split(s) for s in ["train", "dev", "test"])
    tokenizer = encoders.train_tokenizer(train_x, vocab_size=VOCAB_SIZE, min_frequency=MIN_FREQUENCY)
    index, counts = count_matrix(tokenizer, train_x + dev_x + test_x)

    results, rows = {}, []
    combos = [(e, m) for m in MODELS for e in ENCODERS]
    for r, (enc_name, model_name) in enumerate(combos):
        print(f"\n[{fmt(time.time() - start)}] {r + 1}/{len(combos)}: {enc_name} + {model_name}...")
        encoder = make_encoder(enc_name, index, counts, train_x)
        y_pred = fit_predict(model_name, list(zip(train_x, train_y)), test_x, encoder,
                             validation_set=list(zip(dev_x, dev_y)))
        acc, f1 = accuracy_score(test_y, y_pred), f1_score(test_y, y_pred, average="macro")
        results[f"{enc_name}+{model_name}"] = {"test_accuracy": acc, "test_macro_f1": f1}
        with open(os.path.join(RESULTS_DIR, "final_test.json"), "w") as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(RESULTS_DIR, f"cv_{enc_name}_{model_name}.json")) as f:
            folds = json.load(f)["folds"]
        rows.append([f"{enc_name}+{model_name}", acc, np.mean([x["accuracy"] for x in folds]),
                     f1, np.mean([x["macro_f1"] for x in folds])])
        print(f"[{fmt(time.time() - start)}] test accuracy {acc:.4f}, macro-F1 {f1:.4f}")

    print("\n=== Test split (getraind op train, checkpoint gekozen op dev) ===")
    print(tabulate(sorted(rows, key=lambda x: -x[1]), floatfmt=".4f", tablefmt="github",
                   headers=["systeem", "test acc", "CV acc", "test macro-F1", "CV macro-F1"]))


if __name__ == "__main__":
    main()
