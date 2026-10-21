"""Structure-only probe: a classifier on text-only citation features (structure_features.py), evaluated on the same
rows and the same 10x10 grouped folds as analysis_pruning.py (rows in loader order, df.sample(frac=1, random_state=42+rep),
StratifiedGroupKFold(10, shuffle, random_state=42+rep)), with the same class balancing and Youden threshold.
--holdout checks that the re-created folds equal those of the main run. Writes <out>/<cfg>/rep<r>_fold_scores.csv in
the fold_scores.csv layout. Also evaluates the trivial baselines random_5050, proportional and all_halluc.
"""
import sys, os, json, glob, argparse, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from interpret.glassbox import ExplainableBoostingClassifier
from analysis_pruning import youden_threshold_oof
STRUCT = ["stmt_log_chars", "stmt_log_tokens", "n_fullstops", "is_first_cite", "seen_before", "same_as_last", "n_prev_cites", "n_in_bracket", "k_in_bracket",
          "n_num_words", "prop_num_words", "n_cap_words", "prop_cap_words", "doc_log_chars", "doc_rel_pos", "cited_num"]
ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True, help="Name of this configuration (output sub-directory).")
ap.add_argument("--results_file", required=True); ap.add_argument("--reps", type=int, default=10); ap.add_argument("--struct_feats", default=""); ap.add_argument("--struct_side", default=""); ap.add_argument("--combo_name", default="Struct-v2"); ap.add_argument("--only_struct", action="store_true"); ap.add_argument("--rep_from", type=int, default=0); ap.add_argument("--force", action="store_true"); ap.add_argument("--n_jobs", type=int, default=8)
ap.add_argument("--holdout", default="", help="rep0 holdout_predictions.csv of the main run: verifies that the re-created folds are identical"); ap.add_argument("--out", required=True); ap.add_argument("--max_missing_frac", type=float, default=0.002)
ap.add_argument("--models", choices=["lr", "all", "tuned"], default="lr", help="lr: untuned L2 logistic regression; all: + LightGBM and EBM; tuned: L2 / elastic-net logistic regression and ridge with the penalty tuned by inner cross-validation.")
ap.add_argument("--group_by", choices=["topic", "report"], default="topic")
a = ap.parse_args()
if a.struct_feats: STRUCT = a.struct_feats.replace('+', ',').split(',')
res_path = a.results_file
by = collections.defaultdict(list)
with open(res_path) as f:
    for line in f:
        if '"source_id"' not in line: continue
        try: d = json.loads(line)
        except Exception: continue
        sid = d.get("source_id")
        if sid: by[sid].append((d.get("id"), [(t["token_index"], t["token_str"], t["label"]) for t in d["token_data"]]))
rows = []
for sid in sorted(by):
    for rid, toks in by[sid]:
        for ti, ts, lab in toks: rows.append((sid, "%s|%s" % (sid, rid), ti, lab))
base = pd.DataFrame(rows, columns=["source_id", "key", "token_index", "label"]); del by, rows
base["group"] = base["source_id"].map(lambda sid: str(sid).split("_")[0].split("~")[0]) if a.group_by == "topic" else base["source_id"]
print("[eval] rows in score file:", len(base), flush=True)
side = json.load(open(a.struct_side))
vals = [side.get("%s|%d" % (k_, t_)) for k_, t_ in zip(base["key"], base["token_index"])]
cov = float(np.mean([v is not None for v in vals])); print("[eval] feature coverage: %.2f%% of rows" % (100 * cov), flush=True)
if cov < 0.97: print("[eval] ABORT: coverage too low"); sys.exit(2)
for c in STRUCT: base[c] = pd.to_numeric(pd.Series([v.get(c) if v is not None else None for v in vals]), errors="coerce").to_numpy()
base["oor"] = [(v.get("oor") if v is not None else None) for v in vals]
for c in STRUCT: base[c] = base[c].fillna(base[c].median()).fillna(0.0)
for c in ("ppl", "ln_entropy", "energy", "ptrue"): base[c] = np.nan

def metrics(y, score, pred, flip_pcc=False):
    out = {}
    out["AUC"] = roc_auc_score(y, score) if len(np.unique(y)) > 1 and np.ptp(score) > 0 else 0.5
    if np.ptp(score) > 0 and len(np.unique(y)) > 1:
        out["PCC"] = pearsonr(y, score)[0] * (-1 if flip_pcc else 1)
    else: out["PCC"] = 0.0
    out["Precision"] = precision_score(y, pred, zero_division=0); out["Recall"] = recall_score(y, pred, zero_division=0); out["F1"] = f1_score(y, pred, zero_division=0)
    return out
def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    if len(thr) == 0: return 0.5
    return thr[int(np.argmax(tpr - fpr))]
def run_fold(df, rep, fold_id, tr, te, seed):
    res = []
    train, test = df.iloc[tr], df.iloc[te]; y_te = test["label"].to_numpy().astype(int)
    if len(np.unique(y_te)) < 2: return res
    cls0, cls1 = train[train["label"] == 0], train[train["label"] == 1]
    if len(cls0) == 0 or len(cls1) == 0: return res
    n_bal = min(len(cls0), len(cls1))
    c0 = resample(cls0, replace=False, n_samples=n_bal, random_state=seed) if len(cls0) > n_bal else cls0
    c1 = resample(cls1, replace=False, n_samples=n_bal, random_state=seed) if len(cls1) > n_bal else cls1
    bal = pd.concat([c1, c0]); rng = np.random.default_rng(1000 * rep + fold_id)
    def add(clf, combo, m):
        if a.only_struct and combo != a.combo_name: return
        for k, v in m.items(): res.append(dict(classifier=clf, combo=combo, metric=k, fold=fold_id, value=float(v)))
    n = len(te); prev = train["label"].mean()
    add("-", "random_5050", metrics(y_te, rng.random(n), (rng.random(n) < 0.5).astype(int)))
    add("-", "proportional", metrics(y_te, rng.random(n), (rng.random(n) < prev).astype(int)))
    add("-", "all_halluc", metrics(y_te, np.zeros(n), np.ones(n, dtype=int)))
    for name, col in (("Perplexity", "ppl"), ("LN-Entropy", "ln_entropy"), ("Energy", "energy")):
        ok_te = test[col].notna().to_numpy(); ok_tr = bal[col].notna().to_numpy()
        if ok_te.sum() < 5 or len(np.unique(y_te[ok_te])) < 2 or len(np.unique(bal["label"].to_numpy()[ok_tr])) < 2: continue
        thr = youden(bal["label"].to_numpy()[ok_tr], bal[col].to_numpy()[ok_tr]); s = test[col].to_numpy()[ok_te]
        add("-", name, metrics(y_te[ok_te], s, (s > thr).astype(int)))
    ok = test["ptrue"].notna().to_numpy()
    if ok.sum() >= 5 and len(np.unique(y_te[ok])) > 1:
        s = test["ptrue"].to_numpy()[ok]; add("-", "P(True)", metrics(y_te[ok], 1 - s, (s < 0.5).astype(int), flip_pcc=False) | {"PCC": pearsonr(y_te[ok], s)[0] * -1})
    if a.models == "tuned":
        from sklearn.base import BaseEstimator, ClassifierMixin
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import StratifiedGroupKFold as SGK, cross_val_score
        class RidgeProb(BaseEstimator, ClassifierMixin):
            """ridge regression on the 0/1 label (linear probability model); predict_proba = clipped prediction"""
            def __init__(self, alpha=1.0): self.alpha = alpha
            def fit(self, X, y):
                self.classes_ = np.array([0, 1]); self.m_ = Ridge(alpha=self.alpha).fit(X, y); return self
            def predict_proba(self, X):
                p = np.clip(self.m_.predict(X), 0.0, 1.0); return np.column_stack([1 - p, p])
            def predict(self, X): return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
        def pick(factory, grid, X, y, g):
            best = None
            for c in grid:
                m_ = factory(c)
                try: sc = cross_val_score(m_, X, y, groups=g, cv=SGK(n_splits=5, shuffle=True, random_state=seed), scoring="roc_auc").mean()
                except Exception: sc = -1
                if best is None or sc > best[0]: best = (sc, m_)
            return best[1]
        X_, y_, g_ = bal[STRUCT], bal["label"], bal["group"]
        lr2 = lambda C: Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(penalty="l2", C=C, max_iter=2000, random_state=seed, class_weight="balanced"))])
        en = lambda C: Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=C, max_iter=3000, random_state=seed, class_weight="balanced"))])
        rg = lambda al: Pipeline([("scaler", StandardScaler()), ("classifier", RidgeProb(alpha=al))])
        models = {"LR-L2 (tuned C)": pick(lr2, (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0), X_, y_, g_),
                  "LR elastic net (tuned C)": pick(en, (0.01, 0.03, 0.1, 0.3, 1.0), X_, y_, g_),
                  "Ridge regression (linear prob.)": pick(rg, (1.0, 10.0, 100.0, 1000.0, 10000.0), X_, y_, g_)}
    else:
        models = {"Logistic Regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"))]),
                  "LightGBM": lgb.LGBMClassifier(random_state=seed, class_weight="balanced", verbose=-1, n_jobs=1),
                  "EBM": ExplainableBoostingClassifier(random_state=seed, interactions=0, n_jobs=1)}
        if a.models == "lr": models = {"Logistic Regression": models["Logistic Regression"]}
    Xb, yb = bal[STRUCT], bal["label"]; Xt = test[STRUCT]
    from sklearn.base import clone
    for mn, m in models.items():
        cm = clone(m).fit(Xb, yb); pr = cm.predict_proba(Xt)[:, 1]
        thr = youden_threshold_oof(m, Xb, yb, bal["group"], seed)
        add(mn, a.combo_name, metrics(y_te, pr, (pr > thr).astype(int)))
    return res
os.makedirs("%s/%s" % (a.out, a.cfg), exist_ok=True)
for rep in range(a.rep_from, a.reps):
    seed = 42 + rep
    if os.path.exists("%s/%s/rep%d_fold_scores.csv" % (a.out, a.cfg, rep)) and not a.force: print("[eval] rep", rep, "exists, skipping", flush=True); continue
    df = base.sample(frac=1, random_state=seed).reset_index(drop=True)
    y = df["label"].astype(int); g = df["group"]
    sg = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    splits = list(sg.split(df, y, g))
    if rep == a.rep_from and a.holdout:
        h = pd.read_csv(a.holdout.replace("rep0", "rep%d" % rep)); h = h[(h["classifier"] == h["classifier"].iloc[0]) & (h["combo"] == "Baseline")]
        mine = pd.concat([df.iloc[te][["source_id", "token_index", "label"]] for tr, te in splits if df.iloc[te]["label"].nunique() > 1]).reset_index(drop=True)
        same = len(h) == len(mine) and (h["source_id"].astype(str).to_numpy() == mine["source_id"].astype(str).to_numpy()).all() and (h["token_index"].to_numpy() == mine["token_index"].to_numpy()).all() and (h["label"].to_numpy() == mine["label"].to_numpy()).all()
        print("[eval] FOLD-IDENTITY CHECK vs main run holdout (rep0): rows main=%d mine=%d identical=%s" % (len(h), len(mine), same), flush=True)
        if not same: print("[eval] ABORT: folds differ"); sys.exit(3)
    out = Parallel(n_jobs=a.n_jobs)(delayed(run_fold)(df, rep, i, tr, te, seed) for i, (tr, te) in enumerate(splits))
    rows = [r for o in out for r in o]
    pd.DataFrame(rows).to_csv("%s/%s/rep%d_fold_scores.csv" % (a.out, a.cfg, rep), index=False)
    print("[eval] rep", rep, "done:", len(rows), "metric rows", flush=True)
print("[eval] DONE", a.cfg)
