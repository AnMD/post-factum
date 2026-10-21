"""
Directional relationship of each FACTUM score with correct citations, from the held-out aggregate features dumped
by analysis_pruning.py --dump_agg (agg_EBM_fold*.csv, per-layer aggregation). For BAS, CAS, PAS and PFS and for the
mean and the max over layers (head-mean per layer first for BAS and CAS), the effect is the probability that a random
correct citation has a higher value than a random hallucinated one (ties count 0.5). The p-value is a two-sided
bootstrap test that resamples reports (B = 2000, RNG seeded from --name).

usage: directional_bootstrap.py --dump_dir D --name NAME --out cells.json
"""
import argparse, glob, json, zlib
import numpy as np, pandas as pd
from scipy.stats import rankdata

CONCEPTS = [("BAS", "bos_attention_proportion_per_head"), ("CAS", "cas_alt_vs_final_layer_per_head"),
            ("PAS", "pos_cos_similarity_per_layer"), ("PFS", "v_ffn_norm_per_layer")]
B = 2000
ap = argparse.ArgumentParser()
ap.add_argument("--dump_dir", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--min_folds", type=int, default=8)
a = ap.parse_args()
rng = np.random.default_rng(zlib.crc32(a.name.encode()))
def effect(x, y):
    r = rankdata(x); n0 = (y == 0).sum(); n1 = len(y) - n0
    return np.nan if n0 == 0 or n1 == 0 else (r[y == 0].sum() - n0 * (n0 + 1) / 2) / (n0 * n1)
def boot_p(x, y, g):
    ug, inv = np.unique(g, return_inverse=True); idx = [np.where(inv == i)[0] for i in range(len(ug))]; out = []
    for _ in range(B):
        pick = rng.integers(0, len(ug), len(ug)); ii = np.concatenate([idx[i] for i in pick]); e = effect(x[ii], y[ii])
        if not np.isnan(e): out.append(e)
    out = np.array(out); return 2 * min((out <= 0.5).mean(), (out >= 0.5).mean())
cells = {}
fs = sorted(glob.glob("%s/agg_EBM_fold*.csv" % a.dump_dir))
if len(fs) < a.min_folds: raise SystemExit("only %d fold dumps in %s" % (len(fs), a.dump_dir))
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
for sh, prefix in CONCEPTS:
    for ag in ("mean", "max"):
        f = next((("%s_%s_%s" % (prefix, lv, ag)) for lv in ("lmean", "lval") if ("%s_%s_%s" % (prefix, lv, ag)) in df.columns), None)
        m = df[f].notna(); x = df.loc[m, f].to_numpy(float); y = df.loc[m, "label"].to_numpy(int); g = df.loc[m, "source_id"].astype(str).to_numpy()
        cells["|".join((a.name, sh, ag))] = dict(effect=float(effect(x, y)), p=float(boot_p(x, y, g)), feat=f, n0=int((y == 0).sum()), n1=int((y == 1).sum()), nfolds=len(fs))
json.dump(cells, open(a.out, "w"), indent=1); print("ok", a.name, len(cells), "cells")
