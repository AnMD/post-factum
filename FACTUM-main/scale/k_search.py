"""
Choice of the pruning level k and the reporting runs.

  k_search.py commands --results_file F --out_dir D [--grid 5:100:5] [--search_seeds 100,101,102]
      prints one analysis_pruning.py command per (k, seed): --n_repeats 1, LogReg and LightGBM, per-layer aggregation.
  k_search.py select --out_dir D [--grid 5:100:5] [--search_seeds 100,101,102] [--json chosen_k.json]
      selection score of k = mean fold AUC over {Logistic Regression, LightGBM} x {Baseline, Ours 1, Ours 2}
      (mean per feature set first, then over the feature sets); the k with the highest score is chosen
      (smallest k on ties).
  k_search.py final --results_file F --out_dir D --k K [--run_ebm] [--reps 10]
      prints the reporting runs at the chosen k (10 runs with --n_repeats 1 and --seed 42+r) and the
      significance.py command that pools their fold_scores.csv.

Run all commands with PYTHONHASHSEED=0 and OMP_NUM_THREADS=1.
"""
import argparse
import collections
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEL_CLF = ("Logistic Regression", "LightGBM")


def grid(spec):
    a, b, s = (int(x) for x in spec.split(":"))
    return list(range(a, b + 1, s))


def fold_file(out_dir, k, seed):
    return os.path.join(out_dir, "k%d" % k, "seed%d" % seed, "ablation_k_%d" % k, "fold_scores.csv")


def score(out_dir, k, seeds):
    files = [fold_file(out_dir, k, s) for s in seeds]
    if not all(os.path.exists(f) for f in files):
        return None
    acc = collections.defaultdict(list)
    for f in files:
        for row in csv.DictReader(open(f)):
            if row["metric"] == "AUC" and row["classifier"] in SEL_CLF and row["value"].strip() not in ("", "nan"):
                acc[row["combo"]].append(float(row["value"]))
    if not acc:
        return None
    per = {c: sum(v) / len(v) for c, v in acc.items()}
    return sum(per.values()) / len(per)


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands")
    c.add_argument("--results_file", required=True)
    c.add_argument("--out_dir", required=True)
    c.add_argument("--grid", default="5:100:5")
    c.add_argument("--search_seeds", default="100,101,102")
    c.add_argument("--extra", default="", help="Extra analysis_pruning.py arguments (e.g. '--group_by report').")
    s = sub.add_parser("select")
    s.add_argument("--out_dir", required=True)
    s.add_argument("--grid", default="5:100:5")
    s.add_argument("--search_seeds", default="100,101,102")
    s.add_argument("--json", default="")
    f = sub.add_parser("final")
    f.add_argument("--results_file", required=True)
    f.add_argument("--out_dir", required=True)
    f.add_argument("--k", type=int, required=True)
    f.add_argument("--reps", type=int, default=10)
    f.add_argument("--run_ebm", action="store_true")
    f.add_argument("--extra", default="")
    a = p.parse_args()
    script = os.path.join(HERE, "analysis_pruning.py")
    if a.cmd == "commands":
        for k in grid(a.grid):
            for seed in (int(x) for x in a.search_seeds.split(",")):
                base = os.path.join(a.out_dir, "k%d" % k, "seed%d" % seed)
                print("python %s --results_file %s --base_dir %s --k_percent %d --n_splits 10 --n_repeats 1 --seed %d --agg_mode perlayer --skip_plots %s"
                      % (script, a.results_file, base, k, seed, a.extra))
    elif a.cmd == "select":
        seeds = [int(x) for x in a.search_seeds.split(",")]
        scores = {k: score(a.out_dir, k, seeds) for k in grid(a.grid)}
        missing = [k for k, v in scores.items() if v is None]
        for k, v in scores.items():
            print("k=%3d  %s" % (k, "%.4f" % v if v is not None else "missing"))
        if missing:
            raise SystemExit("runs missing for k = %s" % missing)
        best = max(sorted(scores), key=lambda k: scores[k])
        print("chosen k = %d" % best)
        if a.json:
            json.dump({"k": best, "scores": scores}, open(a.json, "w"), indent=1)
    else:
        folds = []
        for r in range(a.reps):
            base = os.path.join(a.out_dir, "rep%d" % r)
            print("python %s --results_file %s --base_dir %s --k_percent %d --n_splits 10 --n_repeats 1 --seed %d --agg_mode perlayer --skip_plots%s %s"
                  % (script, a.results_file, base, a.k, 42 + r, " --run_ebm" if a.run_ebm else "", a.extra))
            folds.append(os.path.join(base, "ablation_k_%d" % a.k, "fold_scores.csv"))
        print("python %s --folds %s --out %s" % (os.path.join(HERE, "significance.py"), " ".join(folds), os.path.join(a.out_dir, "significance.csv")))


if __name__ == "__main__":
    main()
