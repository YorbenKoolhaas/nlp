"""
Analyse the cross-validation results written by run_cv.py.

1. Table of mean ± std accuracy and macro-F1 per (encoder, model), best first.
2. Bayesian correlated t-test (Benavoli et al., 2017; baycomp.CorrelatedTTest, with the Nadeau-Bengio
   correction for overlapping training folds) of the best system against every other system, with a ROPE.
   P(best > other), P(practically equivalent), P(other > best).
3. Optional: full pairwise matrix of P(row > col) with --pairwise.

Usage:
    python analyse_cv.py                    # ROPE = 0.01 (1 percentage point)
    python analyse_cv.py --rope 0.005 --metric macro_f1 --pairwise
"""
import argparse
import glob
import json
import os

import baycomp
import numpy as np
from tabulate import tabulate


def load_results(results_dir):
    systems = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "cv_*.json"))):
        with open(path) as f:
            r = json.load(f)
        name = f"{r['config']['encoder']}+{r['config']['model']}"
        systems[name] = {
            "accuracy": np.array([fold["accuracy"] for fold in r["folds"]]),
            "macro_f1": np.array([fold["macro_f1"] for fold in r["folds"]]),
            "config": r["config"],
        }
    return systems


def probs(a, b, rope):
    """P(a > b + rope), P(|a - b| <= rope), P(b > a + rope) from the correlated t-test posterior."""
    p_a, p_rope, p_b = baycomp.CorrelatedTTest(a, b, rope=rope, runs=1).probs()  # diff = b - a; left = a better
    return p_a, p_rope, p_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/vocab8000")
    ap.add_argument("--rope", type=float, default=0.01, help="half-width of the ROPE, in metric units (0.01 = 1%%)")
    ap.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--pairwise", action="store_true", help="also print the full pairwise P(row > col) matrix")
    args = ap.parse_args()

    systems = load_results(args.results_dir)
    if not systems:
        print(f"Geen resultaten in {args.results_dir}/ -- eerst run_cv.py draaien.")
        return
    folds = {len(s["accuracy"]) for s in systems.values()}
    if len(folds) != 1:
        print(f"Let op: systemen hebben verschillende aantallen folds ({folds}); vergelijk alleen gelijke runs.")

    m = args.metric
    ranking = sorted(systems, key=lambda n: -systems[n][m].mean())

    print(f"\n=== Cross-validation ({folds.pop()} folds), gesorteerd op {m} ===")
    print(tabulate(
        [[n, f"{systems[n]['accuracy'].mean():.4f} ± {systems[n]['accuracy'].std(ddof=1):.4f}",
          f"{systems[n]['macro_f1'].mean():.4f} ± {systems[n]['macro_f1'].std(ddof=1):.4f}"] for n in ranking],
        headers=["systeem (encoder+model)", "accuracy", "macro-F1"], tablefmt="github"))

    best = ranking[0]
    print(f"\n=== Bayesiaanse correlated t-test: {best} tegen de rest ({m}, ROPE ±{args.rope}) ===")
    rows = []
    for other in ranking[1:]:
        p_best, p_rope, p_other = probs(systems[best][m], systems[other][m], args.rope)
        diff = systems[best][m].mean() - systems[other][m].mean()
        verdict = ("best is beter" if p_best > 0.95 else
                   "praktisch gelijk" if p_rope > 0.95 else
                   "andere is beter" if p_other > 0.95 else "onbeslist")
        rows.append([other, f"{diff:+.4f}", f"{p_best:.3f}", f"{p_rope:.3f}", f"{p_other:.3f}", verdict])
    print(tabulate(rows, headers=["tegen", "verschil", f"P({best} beter)", "P(ROPE)", "P(ander beter)", "oordeel (>0.95)"],
                   tablefmt="github"))

    if args.pairwise:
        print(f"\n=== P(rij beter dan kolom + ROPE) voor {m}, ROPE ±{args.rope} ===")
        matrix = [[a] + ["-" if a == b else f"{probs(systems[a][m], systems[b][m], args.rope)[0]:.2f}" for b in ranking]
                  for a in ranking]
        print(tabulate(matrix, headers=[""] + ranking, tablefmt="github"))


if __name__ == "__main__":
    main()
