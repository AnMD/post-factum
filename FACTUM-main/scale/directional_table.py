"""
Renders the directional table from the directional_bootstrap.py outputs: BH correction over all cells of the table;
blue up-arrow = higher scores for correct citations, orange down-arrow = lower, shade by BH-adjusted p (5% / 1% / 0.1%).

usage: directional_table.py <cells_dir> <out_prefix>   -> <out_prefix>.tex and <out_prefix>.txt
"""
import json, glob, sys
from statsmodels.stats.multitest import multipletests
CELLS_DIR, OUT = sys.argv[1], sys.argv[2]
cells={}
for f in sorted(glob.glob(CELLS_DIR + "/*.json")): cells.update({tuple(k.split("|")): v for k,v in json.load(open(f)).items()})
keys=list(cells); adj=dict(zip(keys, multipletests([cells[k]["p"] for k in keys], method="fdr_bh")[1]))
def shade(q, up):
    lvl = 3 if q < 0.001 else (2 if q < 0.01 else (1 if q < 0.05 else 0))
    if lvl == 0: return "--"
    return r"\cellcolor{%s!%s}$\%s$" % ("CorrBlue" if up else "HalluOrange", {3: "85", 2: "55", 1: "28"}[lvl], "uparrow" if up else "downarrow")
COLS = [("l8b","Llama"),("q7bfd","Qwen"),("rnatived","Cmd. R"),("h8b","Llama"),("hq7bd","Qwen"),("hr7b","Cmd. R"),("l3b","Llama"),("q3bfd","Qwen"),("h3b","Llama"),("hq3bd","Qwen"),("hplr","Pleias")]
def row(sh): return " & ".join(shade(adj[(c,sh,ag)], cells[(c,sh,ag)]["effect"]>0.5) for c,_ in COLS for ag in ("mean","max"))
L=[r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\label{tab:score-directions}", r"\makebox[\textwidth][c]{%", r"\begin{tabular}{l *{22}{>{\centering\arraybackslash}p{1.75em}}}", r"\toprule",
 r" & \multicolumn{12}{c}{\textbf{8B / 7B}} & \multicolumn{10}{c}{\textbf{3B / 1B}} \\", r"\cmidrule(lr){2-13} \cmidrule(lr){14-23}",
 r" & \multicolumn{6}{c}{NeuCLIR} & \multicolumn{6}{c}{HotpotQA} & \multicolumn{4}{c}{NeuCLIR} & \multicolumn{6}{c}{HotpotQA} \\", r"\cmidrule(lr){2-7} \cmidrule(lr){8-13} \cmidrule(lr){14-17} \cmidrule(lr){18-23}",
 "Score & " + " & ".join(r"\multicolumn{2}{c}{%s}" % n for _, n in COLS) + r" \\", " ".join(r"\cmidrule(lr){%d-%d}" % (2+2*i, 3+2*i) for i in range(11)), r"\midrule"]
for sh in ("BAS","CAS","PAS","PFS"): L.append(r"\textbf{%s} & %s \\" % (sh,row(sh)))
L+=[r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
txt="\n".join(L)
for ext in ("tex","txt"): open("%s.%s"%(OUT,ext),"w").write(txt)
print("written; cells",len(keys))
import collections
for ds,al in (("NeuCLIR",["l8b","q7bfd","rnatived","l3b","q3bfd"]),("HotpotQA",["h8b","hq7bd","hr7b","h3b","hq3bd","hplr"])):
    for sh in ("BAS","CAS","PAS","PFS"):
        c=collections.Counter()
        for a in al:
            for ag in ("mean","max"):
                k=(a,sh,ag); q=adj[k]; c["up" if q<0.05 and cells[k]["effect"]>0.5 else ("down" if q<0.05 else "ns")]+=1
        print(ds,sh,dict(c))
