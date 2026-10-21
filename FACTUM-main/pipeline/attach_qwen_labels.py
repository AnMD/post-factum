"""Attaches judge labels to scores computed on unlabelled citations (good -> 0, bad -> 1). Scored tokens are aligned
with the citations of the scorer input by order (digits must match) to recover each token's character start; the
judge label is looked up by (source_id, start). Tokens whose citation the judge dropped (ties) are removed.
--dry only reports coverage.
"""
import json, argparse, sys
from collections import defaultdict

def load_jsonl(p):
    out=[]
    with open(p) as f:
        for l in f:
            l=l.strip()
            if not l: continue
            try: out.append(json.loads(l))
            except Exception: pass
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scores_<arm>_all.jsonl (source_id + token_data)")
    ap.add_argument("--scoring_in", nargs="+", required=True, help="the response_*.jsonl the extractor consumed (citation order + start)")
    ap.add_argument("--judge", nargs="+", required=True, help="judge response_*.jsonl (labels with start + label_type)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true")
    a=ap.parse_args()

    scores={str(d["source_id"]): d for d in load_jsonl(a.scores)}
    sin=defaultdict(list)
    for f in a.scoring_in:
        for d in load_jsonl(f):
            sin[str(d["source_id"])] = d.get("labels", [])
    jlab={}
    for f in a.judge:
        for d in load_jsonl(f):
            sid=str(d["source_id"])
            for lb in d.get("labels", []):
                jlab[(sid, int(lb["start"]))] = lb["label_type"]

    n_tok=n_good=n_bad=n_nolabel=n_align_fail=0
    labeled=[]
    for sid, sd in scores.items():
        toks=sd["token_data"]; cits=sin.get(sid, [])
        out_toks=[]
        ti=0
        for cit in cits:
            target=str(cit.get("text",""))
            grp=[]; acc=""
            while ti < len(toks) and acc != target:
                acc += str(toks[ti].get("token_str",""))
                grp.append(toks[ti]); ti+=1
                if len(acc) > len(target):
                    break
            if acc != target:
                n_align_fail += len(grp)
                continue
            lt = jlab.get((sid, int(cit["start"])))
            for t in grp:
                n_tok+=1
                if lt=="good":
                    t=dict(t); t["label"]=0; out_toks.append(t); n_good+=1
                elif lt=="bad":
                    t=dict(t); t["label"]=1; out_toks.append(t); n_bad+=1
                else:
                    n_nolabel+=1
        if out_toks:
            labeled.append({"source_id": sid, "token_data": out_toks})

    print(f"[attach] sids={len(scores)} tokens_labeled={n_tok} good={n_good} bad={n_bad} "
          f"dropped(no-judge-label/tie)={n_nolabel} | align_fail_tokens={n_align_fail}")
    if a.dry:
        print("[attach] DRY RUN — nothing written.")
        return
    with open(a.out,"w") as f:
        for r in labeled: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"[attach] wrote {len(labeled)} labeled sids -> {a.out}")

if __name__=="__main__":
    main()
