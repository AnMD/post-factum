"""Figure of the k search: average AUC per pruning level k (mean over the fold x seed values of the selection score of
k_search.py, i.e. the mean fold AUC over {Logistic Regression, LightGBM} x {Baseline, Ours 1, Ours 2}), with
+-1 Nadeau-Bengio corrected standard error bands (SD * sqrt(1/n + 1/9)) and a filled dot at the chosen k.

The config is a JSON list of panels:
  [{"title": "8B / 7B models",
    "models": [{"name": "Llama-3.1-8B", "color": "#2b6cb0", "neuclir24": "<k_search.py out_dir>", "hotpotqa": "<out_dir>"}, ...]},
   ...]
A model without a dataset leaves the key out. NeuCLIR curves are solid, HotpotQA curves dashed.
usage: k_curves.py --config panels.json --out k_curves.pdf [k_curves.png ...] [--json curves.json]
"""
import argparse, collections, csv, glob, json, os, re, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from k_search import SEL_CLF, fold_file

XLABEL = "Hyperparameter $k$ (% of features kept post-pruning)"
YLABEL = "Avg AUC"
FIGSIZE = (7.2, 1.81)
YLIM = (0.47, 0.73)
XTICKS = [5, 20, 40, 60, 80, 100]
LINESTYLES = {"neuclir24": "-", "hotpotqa": "--"}
DS_LABEL = {"neuclir24": "NeuCLIR", "hotpotqa": "HotpotQA"}
FONTSIZE = 7
LEGEND_FONTSIZE = 5.8


def curve(out_dir):
    """{k: (mean, SD, n)} over the per-fold selection scores of every complete k, and the chosen k"""
    ks = sorted(int(re.sub(r"\D", "", os.path.basename(p))) for p in glob.glob(os.path.join(out_dir, "k*")))
    seeds = sorted({int(os.path.basename(p)[4:]) for p in glob.glob(os.path.join(out_dir, "k*", "seed*"))})
    res = {}
    for k in ks:
        files = [fold_file(out_dir, k, s) for s in seeds]
        if not all(os.path.exists(f) for f in files):
            continue
        vals, per = [], collections.defaultdict(list)
        for f in files:
            fold = collections.defaultdict(list)
            for row in csv.DictReader(open(f)):
                if row["metric"] == "AUC" and row["classifier"] in SEL_CLF and row["value"].strip() not in ("", "nan"):
                    fold[row["fold"]].append(float(row["value"])); per[row["combo"]].append(float(row["value"]))
            vals += [sum(v) / len(v) for v in fold.values()]
        score = sum(sum(v) / len(v) for v in per.values()) / len(per)
        res[k] = (statistics.mean(vals), statistics.stdev(vals), len(vals), score)
    chosen = max(sorted(res), key=lambda k: res[k][3])
    return {k: v[:3] for k, v in res.items()}, chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, nargs="+", help="Output figure file(s), e.g. k_curves.pdf k_curves.png")
    ap.add_argument("--json", default="", help="Also write the curve values to this file.")
    a = ap.parse_args()
    panels = json.load(open(a.config))
    data = {}
    plt.rcParams.update({"font.size": FONTSIZE, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, axs = plt.subplots(1, len(panels), figsize=FIGSIZE, sharey=True, constrained_layout=True)
    for ax, panel in zip(axs, panels):
        for m in panel["models"]:
            col = m["color"]
            for ds in ("neuclir24", "hotpotqa"):
                if not m.get(ds):
                    continue
                d, ck = curve(m[ds])
                data["%s|%s" % (m["name"], ds)] = {"chosen": ck, "k": {str(k): list(v) for k, v in d.items()}}
                ks = sorted(d)
                mean = [d[k][0] for k in ks]
                se = [d[k][1] * (1 / d[k][2] + 1 / 9) ** 0.5 for k in ks]
                ax.fill_between(ks, [x - s for x, s in zip(mean, se)], [x + s for x, s in zip(mean, se)], color=col, alpha=0.10, lw=0)
                ax.plot(ks, mean, LINESTYLES[ds], color=col, lw=1.1, marker="o", ms=1.9, mfc="white", mew=0.6)
                ax.plot([ck], [d[ck][0]], "o", color=col, ms=4.6, mec="black", mew=0.5, zorder=5)
        ax.set_title(panel["title"], fontsize=FONTSIZE + 0.5, pad=3)
        ax.set_xlabel(XLABEL)
        ax.set_xticks(XTICKS)
        ax.grid(alpha=0.25, lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        handles = [Line2D([0], [0], color=m["color"], lw=1.4, label=m["name"]) for m in panel["models"]]
        handles += [Line2D([0], [0], color="0.3", ls=ls, lw=1.1, label=DS_LABEL[ds]) for ds, ls in LINESTYLES.items()]
        ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper right", ncol=2, handlelength=1.8, columnspacing=0.8, borderaxespad=0.2)
    axs[0].set_ylabel(YLABEL)
    axs[0].set_ylim(*YLIM)
    for f in a.out:
        fig.savefig(f, dpi=250, bbox_inches="tight")
    if a.json:
        json.dump(data, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
