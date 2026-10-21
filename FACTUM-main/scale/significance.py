"""
Significance tests on the per-fold scores written by analysis_pruning.py (fold_scores.csv).

The folds of repeated cross-validation share training data, so the naive paired t-test over
all folds underestimates the variance. We use the Nadeau-Bengio corrected resampled t-test:
    t = mean(d) / sqrt((1/J + n_test/n_train) * var(d)),  df = J - 1,  n_test/n_train = 1/(k-1)
where d are the per-fold differences (comparison - baseline) pooled over all folds of all
repeats and J is the number of folds with both scores. The test is two-sided, and
Benjamini-Hochberg correction is applied per classifier over all comparisons and metrics.
A comparison is marked significant (Star_NB) when it is BH-significant and the comparison
mean is higher than the baseline mean.

For reference, the naive two-sided paired t-test (P_naive_*) and the test used in the original
FACTUM code (one-sided naive paired t-test with BH, P_paper_BH / Star_paper) are also reported.

Usage: python significance.py --folds rep0/fold_scores.csv rep1/fold_scores.csv ... --out results.csv
(one fold_scores.csv per repeat, i.e. per run of analysis_pruning.py with --n_repeats 1)
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

METRICS = ["AUC", "PCC", "Precision", "Recall", "F1"]
COMPARISONS = [("Baseline", "Ours 1"), ("Baseline", "Ours 2"), ("Ours 1", "Ours 2")]
COLUMNS = [
    "Classifier",
    "Comparison",
    "Metric",
    "J",
    "Mean_Baseline",
    "Std_Baseline",
    "Mean_Comparison",
    "Std_Comparison",
    "Mean_Diff",
    "t_naive",
    "P_naive_2sided",
    "P_naive_BH",
    "Star_naive",
    "P_paper_BH",
    "Star_paper",
    "t_NB",
    "P_NB_raw",
    "P_NB_BH",
    "Star_NB",
]


def load_fold_scores(paths):
    """(classifier, combo, metric) -> {(repeat, fold): value}, one file per repeat."""
    scores = defaultdict(dict)
    for repeat, path in enumerate(paths):
        for row in csv.DictReader(open(path)):
            value = (
                float(row["value"])
                if row["value"] not in ("", "nan", "NaN")
                else float("nan")
            )
            scores[(row["classifier"], row["combo"], row["metric"])][
                (repeat, int(row["fold"]))
            ] = value
    return scores


def nadeau_bengio_test(diff, ratio):
    """Returns (t_NB, p_NB two-sided, t_naive, p_naive two-sided, p_naive one-sided 'greater')."""
    diff = np.asarray(diff, float)
    J = len(diff)
    if J < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    mean, var = diff.mean(), diff.var(ddof=1)
    if var <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    t_nb = mean / np.sqrt((1.0 / J + ratio) * var)
    p_nb = 2 * stats.t.sf(abs(t_nb), J - 1)
    t_naive = mean / np.sqrt(var / J)
    p_naive = 2 * stats.t.sf(abs(t_naive), J - 1)
    return t_nb, p_nb, t_naive, p_naive, stats.t.sf(t_naive, J - 1)


def benjamini_hochberg(p_values):
    """BH over the non-NaN p-values; returns (corrected p-values, significant flags)."""
    corrected = [np.nan] * len(p_values)
    significant = [False] * len(p_values)
    valid = [i for i, p in enumerate(p_values) if not np.isnan(p)]
    if valid:
        reject, p_corrected, _, _ = multipletests(
            [p_values[i] for i in valid], alpha=0.05, method="fdr_bh"
        )
        for j, i in enumerate(valid):
            corrected[i] = p_corrected[j]
            significant[i] = bool(reject[j])
    return corrected, significant


def run_tests(scores, n_splits):
    ratio = 1.0 / (n_splits - 1)
    rows = []
    for classifier in sorted({key[0] for key in scores}):
        tests = []
        for base, comp in COMPARISONS:
            for metric in METRICS:
                base_scores = scores.get((classifier, base, metric), {})
                comp_scores = scores.get((classifier, comp, metric), {})
                folds = sorted(
                    f
                    for f in (set(base_scores) & set(comp_scores))
                    if base_scores[f] == base_scores[f] and comp_scores[f] == comp_scores[f]
                )
                if len(folds) < 3:
                    continue
                diff = [comp_scores[f] - base_scores[f] for f in folds]
                t_nb, p_nb, t_naive, p_naive, p_paper = nadeau_bengio_test(diff, ratio)
                tests.append(
                    {
                        "Classifier": classifier,
                        "Comparison": "%s vs. %s" % (comp, base),
                        "Metric": metric,
                        "J": len(folds),
                        "Mean_Baseline": np.mean([base_scores[f] for f in folds]),
                        "Mean_Comparison": np.mean([comp_scores[f] for f in folds]),
                        "Std_Baseline": np.std([base_scores[f] for f in folds], ddof=1),
                        "Std_Comparison": np.std([comp_scores[f] for f in folds], ddof=1),
                        "Mean_Diff": np.mean(diff),
                        "t_naive": t_naive,
                        "P_naive_2sided": p_naive,
                        "t_NB": t_nb,
                        "P_NB_raw": p_nb,
                        "P_paper_raw": p_paper,
                    }
                )
        if not tests:
            continue
        p_nb_bh, sig_nb = benjamini_hochberg([t["P_NB_raw"] for t in tests])
        p_naive_bh, sig_naive = benjamini_hochberg([t["P_naive_2sided"] for t in tests])
        p_paper_bh, sig_paper = benjamini_hochberg([t["P_paper_raw"] for t in tests])
        for i, test in enumerate(tests):
            improved = test["Mean_Diff"] > 0
            test["P_paper_BH"] = p_paper_bh[i]
            test["Star_paper"] = "Yes" if sig_paper[i] else "No"
            test["P_NB_BH"] = p_nb_bh[i]
            test["Star_NB"] = "Yes" if (sig_nb[i] and improved) else "No"
            test["P_naive_BH"] = p_naive_bh[i]
            test["Star_naive"] = "Yes" if (sig_naive[i] and improved) else "No"
            rows.append(test)
    return rows


def write_results(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in COLUMNS})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--folds", nargs="+", required=True, help="fold_scores.csv per repeat")
    parser.add_argument("--out", required=True, help="Output CSV.")
    parser.add_argument("--n_splits", type=int, default=10, help="Folds per repeat (k).")
    args = parser.parse_args()
    rows = run_tests(load_fold_scores(args.folds), args.n_splits)
    write_results(rows, args.out)
    print(
        "%d tests, J up to %d, ratio n_test/n_train=%.3f -> %s"
        % (len(rows), max(r["J"] for r in rows), 1.0 / (args.n_splits - 1), args.out)
    )


if __name__ == "__main__":
    main()
