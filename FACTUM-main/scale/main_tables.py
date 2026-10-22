"""Main results tables, one per classifier (Logistic Regression, LightGBM, EBM), from the per-fold scores.

Rows per model: FACTUM, ECS+PKS, CAS+PFS, Structure (structure_probe.py), Perplexity, LN-Entropy, Energy, P(True)
(paper_baselines_eval.py). Columns: AUC, PCC, Precision, Recall, F1 for NeuCLIR 2024 and HotpotQA.
Significance: Nadeau-Bengio corrected paired t-test (two-sided, ratio 1/9) of FACTUM vs. each row on the folds where
both are defined; Benjamini-Hochberg correction per row over its metrics (--family baseline) or over all tests of a
model x dataset x classifier (--family all).

The config is a JSON list of models:
  {"name": "Llama-3.1-8B",
   "neuclir24": {"k": 15, "folds": "<glob of rep*/ablation_k_15/fold_scores.csv>", "folds_ebm": "<optional glob>",
                 "structure": "<structure_probe.py out/cfg dir>", "structure_classifier": "Logistic Regression",
                 "baselines": "<paper_baselines_eval.py out/cfg dir>"},
   "hotpotqa": {...}}
A dataset can be left out (shown as --). Writes main_table_<LR|LightGBM|EBM>.tex and numbers.json to --out_dir.
"""
import argparse, csv, glob, json, os
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

MET = ["AUC", "PCC", "Precision", "Recall", "F1"]
MET_HDR = ["AUC", "PCC", "Prec.", "Rec.", "F1"]
CLFS = [("Logistic Regression", "LR"), ("LightGBM", "LightGBM"), ("EBM", "EBM")]
ROWS = ["FACTUM", "ECS+PKS", "CAS+PFS", "Structure", "Perplexity", "LN-Entropy", "Energy", "P(True)"]
PAPER_BL = ["Perplexity", "LN-Entropy", "Energy", "P(True)"]
DATASETS = [("neuclir24", "NeuCLIR 2024"), ("hotpotqa", "HotpotQA")]


def val(v):
    return float(v) if v not in ("", "nan", "NaN") else float("nan")


def ok(x):
    return x == x


def load_runs(pattern):
    """{(classifier, combo, metric): {(rep, fold): value}} from <run>/rep<r>/ablation_k_<k>/fold_scores.csv"""
    d = {}
    for p in glob.glob(pattern):
        r = int(os.path.normpath(p).split(os.sep)[-3][3:])
        for row in csv.DictReader(open(p)):
            d.setdefault((row["classifier"], row["combo"], row["metric"]), {})[(r, int(row["fold"]))] = val(row["value"])
    return d


def load_flat(directory):
    """{(classifier, combo, metric): {(rep, fold): value}} from <dir>/rep<r>_fold_scores.csv"""
    d = {}
    for p in glob.glob(os.path.join(directory, "rep*_fold_scores.csv")):
        r = int(os.path.basename(p).split("_")[0][3:])
        for row in csv.DictReader(open(p)):
            d.setdefault((row["classifier"], row["combo"], row["metric"]), {})[(r, int(row["fold"]))] = val(row["value"])
    return d


def nb_p(diff, ratio=1 / 9):
    diff = np.asarray(diff, float); n = len(diff)
    if n < 3 or diff.var(ddof=1) <= 0:
        return float("nan")
    return float(2 * stats.t.sf(abs(diff.mean() / np.sqrt((1 / n + ratio) * diff.var(ddof=1))), n - 1))


def block(fact, struct, base, clf, struct_key, fact_lr):
    """values[row][metric], stds[row][metric], tests[(row, metric)] = (mean diff FACTUM - row, p, J)"""
    vals, stds, tests = {}, {}, {}
    for m in MET:
        fa = {k: v for k, v in fact.get((clf, "Ours 2", m), {}).items() if ok(v)}
        series = {"ECS+PKS": fact.get((clf, "Baseline", m), {}), "CAS+PFS": fact.get((clf, "Ours 1", m), {}),
                  "Structure": struct.get(struct_key + (m,), {})}
        for b in PAPER_BL:
            series[b] = base.get(("-", b, m), {})
        series = {r: {k: v for k, v in s.items() if ok(v)} for r, s in series.items()}
        ecs = series["ECS+PKS"]
        kfe = sorted(set(fa) & set(ecs)); kce = sorted(set(series["CAS+PFS"]) & set(ecs))
        vals.setdefault("FACTUM", {})[m] = np.mean([fa[k] for k in kfe]) if kfe else float("nan")
        vals.setdefault("ECS+PKS", {})[m] = np.mean([ecs[k] for k in kfe]) if kfe else float("nan")
        vals.setdefault("CAS+PFS", {})[m] = np.mean([series["CAS+PFS"][k] for k in kce]) if kce else float("nan")
        vals.setdefault("Structure", {})[m] = np.mean(list(series["Structure"].values())) if series["Structure"] else float("nan")
        for r in ["FACTUM", "ECS+PKS", "CAS+PFS", "Structure"] + PAPER_BL:
            s = fa if r == "FACTUM" else series[r]
            stds.setdefault(r, {})[m] = float(np.std(list(s.values()), ddof=1)) if len(s) > 1 else float("nan")
        fa_lr = {k: v for k, v in fact_lr.get(("Logistic Regression", "Ours 2", m), {}).items() if ok(v)}
        for r in PAPER_BL:
            keys = sorted(set(series[r]) & set(fa_lr))
            vals.setdefault(r, {})[m] = np.mean([series[r][k] for k in keys]) if keys else float("nan")
        for r in ROWS[1:]:
            keys = sorted(set(series[r]) & set(fa))
            d = [fa[k] - series[r][k] for k in keys]
            tests[(r, m)] = (float(np.mean(d)) if d else float("nan"), nb_p(d), len(keys))
    return vals, stds, tests


def bh(tests, family):
    groups = {}
    for key, (md, p, J) in tests.items():
        if p == p:
            groups.setdefault("all" if family == "all" else key[0], []).append(key)
    out = {}
    for g, keys in groups.items():
        q = multipletests([tests[k][1] for k in keys], method="fdr_bh")[1]
        for k, qq in zip(keys, q):
            out[k] = float(qq)
    return out


def mark(res, row, m):
    if row == "FACTUM":
        return ""
    q = res["p_bh"].get((row, m))
    if q is None or not q < 0.05:
        return ""
    return "$^{-}$" if res["tests"][(row, m)][0] > 0 else "$^{+}$"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="JSON list of models (see above).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--family", choices=["baseline", "all"], default="baseline", help="Benjamini-Hochberg family.")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    models = json.load(open(a.config))

    results = {}
    for mod in models:
        name = mod["name"]
        for ds, _ in DATASETS:
            spec = mod.get(ds)
            if spec is None:
                continue
            struct = load_flat(spec["structure"]); base = load_flat(spec["baselines"])
            struct_key = (spec.get("structure_classifier", "Logistic Regression"), "Struct-v2")
            fa = load_runs(spec["folds"])
            facts = {"Logistic Regression": fa, "LightGBM": fa, "EBM": load_runs(spec["folds_ebm"]) if spec.get("folds_ebm") else fa}
            for clf, _ in CLFS:
                fact = facts[clf]
                n_folds = len(fact.get((clf, "Ours 2", "AUC"), {}))
                if n_folds == 0:
                    results[(name, ds, clf)] = None; continue
                vals, stds, tests = block(fact, struct, base, clf, struct_key, facts["Logistic Regression"])
                results[(name, ds, clf)] = dict(vals=vals, stds=stds, tests=tests, p_bh=bh(tests, a.family), k=spec["k"], n_folds=n_folds)

    for clf, short in CLFS:
        L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\label{tab:repro-both-datasets%s}" % ("" if short == "LR" else "-" + short.lower()),
             r"\begin{tabular}{ll ccccc ccccc}", r"\toprule",
             r"& & \multicolumn{5}{c}{\textbf{NeuCLIR 2024}} & \multicolumn{5}{c}{\textbf{HotpotQA}} \\",
             r"\cmidrule(lr){3-7} \cmidrule(lr){8-12}",
             "Model & Feature Set & " + " & ".join(MET_HDR + MET_HDR) + r" \\", r"\midrule"]
        for mi, mod in enumerate(models):
            name = mod["name"]
            if mi:
                L.append(r"\midrule")
            res = [(ds, results.get((name, ds, clf))) for ds, _ in DATASETS]
            kn, kh = [("%d\\%%" % mod[ds]["k"]) if mod.get(ds) else "--" for ds, _ in DATASETS]
            best = {}
            for ds, r_ in res:
                if r_:
                    for m in MET:
                        vs = [round(r_["vals"][row][m], 2) for row in ROWS if r_["vals"][row][m] == r_["vals"][row][m]]
                        best[(ds, m)] = max(vs) if vs else None
            for ri, row in enumerate(ROWS):
                left = name if ri == 0 else (r"\scriptsize $k=%s$ / $%s$" % (kn, kh) if ri == 1 else "")
                cells = []
                for ds, r_ in res:
                    for m in MET:
                        if not r_ or r_["vals"][row][m] != r_["vals"][row][m]:
                            cells.append("--"); continue
                        v = round(r_["vals"][row][m], 2); t = "%.2f" % v
                        cells.append((r"\textbf{%s}" % t if v == best[(ds, m)] else t) + mark(r_, row, m))
                L.append("%s & %s & %s \\\\" % (left, row, " & ".join(cells)))
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
        open(os.path.join(a.out_dir, "main_table_%s.tex" % short), "w").write("\n".join(L))
    json.dump({"%s|%s|%s" % k: (None if v is None else dict(vals=v["vals"], stds=v["stds"], tests={"%s|%s" % kk: vv for kk, vv in v["tests"].items()},
                                                          p_bh={"%s|%s" % kk: vv for kk, vv in v["p_bh"].items()}, k=v["k"], n_folds=v["n_folds"]))
               for k, v in results.items()}, open(os.path.join(a.out_dir, "numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
