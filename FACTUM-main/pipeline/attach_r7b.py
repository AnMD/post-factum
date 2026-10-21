"""Attaches judge labels to Command R7B / Pleias scores (good -> 0, bad -> 1). Each raw citation is mapped to its
canonical citation (for native citations: the (sentence, document) pair of canonicalize_sentences) and gets the judge
label at that citation's start in the canonical text. Tokens of judge-dropped ties and unaligned tokens are removed.
"""
import argparse, json, re
from collections import defaultdict

CITE_RE = re.compile(r"\[Source:\s*([0-9]+(?:\s*,\s*[0-9]+)*)\]")
DIGIT_RE = re.compile(r"\d+")

def canon_starts(text):
    """char start of every marker digit in `text`, in the order argue_judge.parse_citations emits them."""
    out = []
    for m in CITE_RE.finditer(text):
        inner_start = m.start(1)
        for d in DIGIT_RE.finditer(m.group(1)):
            out.append((int(d.group(0)), inner_start + d.start()))
    return out

def load(p):
    rows = []
    for l in open(p):
        l = l.strip()
        if l:
            try: rows.append(json.loads(l))
            except Exception: pass
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True); ap.add_argument("--gen", required=True)
    ap.add_argument("--judge", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    gens = {str(g["source_id"]): g for g in load(a.gen)}
    scores = {str(d["source_id"]): d for d in load(a.scores)}
    jl = {}
    jsids = set()
    for f in a.judge:
        for d in load(f):
            sid = str(d["source_id"]); jsids.add(sid)
            for lb in d.get("labels", []): jl[(sid, int(lb["start"]))] = lb["label_type"]
    n_tok = n_good = n_bad = n_tie = n_unjudged = n_align_fail = n_struct = 0
    out_rows = []
    for sid, sd in scores.items():
        g = gens.get(sid)
        if g is None: n_struct += 1; continue
        canon = g["responses"][0]; cs = canon_starts(canon)
        cm = g["cit_map"]; native = g["arm"] == "native"
        if native:
            cc = [tuple(x) for x in g["canon_cits"]]; pos_of = {}
            for k, key in enumerate(cc): pos_of.setdefault(key, k)
        if (not native and len(cs) != len(cm)) or (native and len(cs) != len(cc)):
            n_struct += 1; continue
        toks = sd["token_data"]; ti = 0; kept = []
        for i, c in enumerate(cm):
            k = pos_of[(c["sent_idx"], c["canon_n"])] if native else i
            start = cs[k][1]
            lt = jl.get((sid, start))
            target = str(c["raw_idx"]); grp = []; acc = ""
            while ti < len(toks) and acc != target:
                acc += str(toks[ti].get("token_str", "")).strip(); grp.append(toks[ti]); ti += 1
                if len(acc) > len(target): break
            if acc != target: n_align_fail += len(grp); continue
            for t in grp:
                n_tok += 1
                if sid not in jsids: n_unjudged += 1; continue
                if lt == "good": t = dict(t); t["label"] = 0; kept.append(t); n_good += 1
                elif lt == "bad": t = dict(t); t["label"] = 1; kept.append(t); n_bad += 1
                else: n_tie += 1
        if kept: out_rows.append({"source_id": sid, "token_data": kept})
    print("[attach] reports scored=%d judged=%d | tokens: %d -> good=%d bad=%d ties_dropped=%d unjudged=%d | align_fail_tokens=%d struct_skips=%d"
          % (len(scores), len(jsids & set(scores)), n_tok, n_good, n_bad, n_tie, n_unjudged, n_align_fail, n_struct))
    if not a.dry:
        with open(a.out, "w") as f:
            for r in out_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("[attach] wrote %d labeled reports -> %s" % (len(out_rows), a.out))

if __name__ == "__main__":
    main()
