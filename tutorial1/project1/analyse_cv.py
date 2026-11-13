"""
Compare the cross-validation results from run_cv.py with a Bayesian correlated t-test
(Benavoli et al., 2017; baycomp) and a ROPE.

    python analyse_cv.py                              # accuracy, ROPE ±0.01
    python analyse_cv.py --metric macro_f1 --rope 0.005
"""
import argparse
import glob
import json

import baycomp
import numpy as np
from tabulate import tabulate

ap = argparse.ArgumentParser()
ap.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy")
ap.add_argument("--rope", type=float, default=0.01, help="ROPE half-width (0.01 = 1 percentage point)")
args = ap.parse_args()

scores = {}
for path in glob.glob("results/cv_*.json"):
    r = json.load(open(path))
    scores[f"{r['encoder']}+{r['model']}"] = {m: np.array([f[m] for f in r["folds"]]) for m in ["accuracy", "macro_f1"]}
ranking = sorted(scores, key=lambda n: -scores[n][args.metric].mean())

print(f"\n=== 5-fold cross-validation, gesorteerd op {args.metric} ===")
print(tabulate([[n] + [f"{scores[n][m].mean():.4f} ± {scores[n][m].std(ddof=1):.4f}" for m in ["accuracy", "macro_f1"]]
                for n in ranking], headers=["systeem", "accuracy", "macro-F1"], tablefmt="github"))

best = ranking[0]
print(f"\n=== {best} tegen de rest ({args.metric}, ROPE ±{args.rope}) ===")
rows = []
for other in ranking[1:]:
    # diff = other - best; p_left = P(best beter), p_rope = P(praktisch gelijk), p_right = P(other beter)
    p_best, p_rope, p_other = baycomp.CorrelatedTTest(scores[best][args.metric], scores[other][args.metric],
                                                      rope=args.rope).probs()
    verdict = "best beter" if p_best > 0.95 else "praktisch gelijk" if p_rope > 0.95 else \
              "ander beter" if p_other > 0.95 else "onbeslist"
    rows.append([other, f"{scores[best][args.metric].mean() - scores[other][args.metric].mean():+.4f}",
                 f"{p_best:.3f}", f"{p_rope:.3f}", f"{p_other:.3f}", verdict])
print(tabulate(rows, headers=["tegen", "verschil", "P(best beter)", "P(ROPE)", "P(ander beter)", "oordeel (>0.95)"],
               tablefmt="github"))
