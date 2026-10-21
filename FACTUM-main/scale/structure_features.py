"""Text-only structure features for every scored citation token (no model internals): statement length in characters
and tokens, number of full stops, whether this is the first citation / the source was cited before / equals the
previous citation, number of earlier citations, bracket size and position in the bracket, number and proportion of
words with digits and of capitalised non-initial words, length and relative position of the cited document in the
context, and the cited number. The statement is the text between the previous citation bracket and this one.
Output: JSON {"<source_id>|<id>|<token_index>": {feature: value}}; 'oor' (not a feature) marks citations of a
document that is not in the context.
"""
import sys, os, re, json, glob, math, argparse, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
def load(p):
    out = []
    with open(p) as f:
        for l in f:
            l = l.strip()
            if l:
                try: out.append(json.loads(l))
                except Exception: pass
    return out
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="Hugging Face id of the model whose tokenizer counts statement tokens.")
ap.add_argument("--resp", nargs="+", default=[], help="Scorer response files (not needed with --native).")
ap.add_argument("--src", nargs="+", required=True, help="Scorer source_info files.")
ap.add_argument("--scores", required=True, help="Labelled score file whose rows get features.")
ap.add_argument("--out", required=True)
ap.add_argument("--native", action="store_true", help="Command R7B / Pleias: map native citations through the generation file (--gen) and the unlabelled raw scores (--raw_scores).")
ap.add_argument("--gen", default=""); ap.add_argument("--raw_scores", default="")
a = ap.parse_args(); C = {"model": a.model, "resp": a.resp, "src": a.src}
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(C["model"])
BR = re.compile(r"\[[^\]]*?Source[^\]]*\]", re.I)
STD = re.compile(r"(?m)^[ \t]*Source: (\d+)[ \t]*\n"); R7 = re.compile(r'"(\d+)": \{"heading": "Source (\d+):'); PL = re.compile(r"<\|source_id\|>(\d+)")
def docs_of(ctx):
    """-> ordered list of (number, n_chars) of the documents of a context string (3 formats: 'Source: n' blocks, Command-R JSON docs, Pleias source tags)"""
    for rx, g in ((STD, 1), (R7, 2), (PL, 1)):
        ms = list(rx.finditer(ctx))
        if ms: return [(int(m.group(g)), (ms[i + 1].start() if i + 1 < len(ms) else len(ctx)) - m.end()) for i, m in enumerate(ms)]
    return []
def feats(text, cs, docs, tokfn):
    brs = [(m.start(), m.end(), [int(x) for x in re.findall(r"\d+", m.group(0))]) for m in BR.finditer(text)]
    bi = next((i for i, (b0, b1, _) in enumerate(brs) if b0 <= cs < b1), None)
    if bi is None: return None
    b0, b1, nums = brs[bi]; cited = -1; kin = 0
    for q, mm in enumerate(re.finditer(r"\d+", text[b0:b1])):
        if mm.start() <= cs - b0 < mm.end(): cited = int(mm.group(0)); kin = q; break
    prev = brs[bi - 1] if bi > 0 else None
    if prev: pe = prev[1]; consumed = text[pe:pe + 1] == "."; pe += consumed
    else: pe = 0; consumed = True
    s = text[pe:b0]; lead = s[:len(s) - len(s.lstrip(" \n.,;"))]; starts = consumed or "." in lead; s = s.strip()
    words = s.split(); nw = max(len(words), 1); ncap = nnum = 0
    for j, w in enumerate(words):
        init = (j == 0 and starts) or (j > 0 and words[j - 1][-1:] in ".!?")
        c = w.strip(".,;:!?()\"'“”‘’")
        if any(ch.isdigit() for ch in c): nnum += 1
        if c[:1].isupper() and not init: ncap += 1
    seen = {n for _, _, ns_ in brs[:bi] for n in ns_}; last = brs[bi - 1][2][-1] if prev and brs[bi - 1][2] else None
    order = [d[0] for d in docs]; dmap = dict(docs)
    f = {"stmt_log_chars": math.log1p(len(s)), "stmt_log_tokens": math.log1p(len(tokfn(s)) if s else 0), "n_fullstops": len(re.findall(r"[.!?](?=\s|$)", s)),
         "is_first_cite": int(prev is None), "seen_before": int(cited in seen), "same_as_last": int(last is not None and cited == last), "n_prev_cites": sum(len(x[2]) for x in brs[:bi]),
         "n_in_bracket": len(nums), "k_in_bracket": kin, "n_num_words": nnum, "prop_num_words": nnum / nw, "n_cap_words": ncap, "prop_cap_words": ncap / nw, "cited_num": cited}
    if docs:
        if cited in dmap: f["doc_log_chars"] = math.log1p(dmap[cited]); f["doc_rel_pos"] = order.index(cited) / max(len(order) - 1, 1); f["oor"] = 0
        else: f["doc_log_chars"] = None; f["doc_rel_pos"] = None; f["oor"] = 1
    else: f["doc_log_chars"] = None; f["doc_rel_pos"] = None; f["oor"] = None
    return f
tokfn = lambda t: tok(t, add_special_tokens=False).input_ids
out = {}; nrep = nfail = nrows = 0; recs = load(a.scores)
if not a.native:
    resp = {}; srcs = {}
    for p in C["resp"]:
        for r in load(p):
            resp.setdefault(str(r["source_id"]), []).append(r)
            if r.get("id") is not None: resp.setdefault(str(r["id"]), []).append(r)
    for p in C["src"]:
        for r in load(p): srcs[str(r["source_id"])] = r
    def candidates(rec):
        sid = str(rec["source_id"]); rid = rec.get("id"); keys = [sid]
        if rid is not None: keys += [f"{sid}~s{rid}", f"{sid}_s{rid}", str(rid), str(rid).replace("_s", "~s"), f"{sid}_{rid}"]
        seen, o = set(), []
        for k in keys:
            for r in resp.get(k, []):
                if id(r) not in seen: seen.add(id(r)); o.append(r)
        return o
    held = []; plen_by_sid = {}
    for rec in recs:
        rows = [t for t in rec["token_data"] if t.get("label") in (0, 1)]
        if not rows: continue
        key = "%s|%s" % (rec["source_id"], rec.get("id")); got = None
        for cand in candidates(rec):
            text = cand["response"]; enc = tok(text, add_special_tokens=False, return_offsets_mapping=True); offs = enc["offset_mapping"]
            dec = [tok.decode([i]).strip() for i in enc.input_ids]; spans = [(m.start(), m.end()) for m in BR.finditer(text)]
            r0 = rows[0]; ts0 = str(r0["token_str"]).strip(); valids = []
            for j in [j for j, d in enumerate(dec) if d == ts0]:
                plen = r0["token_index"] - j
                if plen <= 0: continue
                ok = True
                for t in rows:
                    i = t["token_index"] - plen
                    if not (0 <= i < len(dec) and dec[i] == str(t["token_str"]).strip() and any(ch.isdigit() for ch in text[offs[i][0]:offs[i][1]]) and any(b0 <= offs[i][1] - 1 < b1 for b0, b1 in spans)): ok = False; break
                if ok: valids.append(plen)
            if valids: got = (rec, key, cand, valids, offs, rows); break
        if got is None: nfail += len(rows); continue
        held.append(got)
        if len(got[3]) == 1: plen_by_sid[str(rec["source_id"])] = got[3][0]
    for rec, key, cand, valids, offs, rows in held:
        plen = plen_by_sid.get(str(rec["source_id"]), valids[0]); plen = plen if plen in valids else valids[0]
        text = cand["response"]; s = srcs.get(str(cand["source_id"])); docs = docs_of(s["context"]) if s else []
        for t in rows:
            o0, o1 = offs[t["token_index"] - plen]; cs_ = o0 + next(i for i, ch in enumerate(text[o0:o1]) if ch.isdigit())
            f = feats(text, cs_, docs, tokfn)
            if f is None: nfail += 1; continue
            out["%s|%d" % (key, t["token_index"])] = f; nrows += 1
        nrep += 1
else:
    import attach_r7b as AR
    GEN, RAW = a.gen, a.raw_scores
    recs = load(RAW)
    gens = {str(g["source_id"]): g for g in AR.load(GEN)}
    srcs = {}
    for p in C["src"]:
        for r in load(p): srcs[str(r["source_id"])] = r
    for sd in recs:
        sid = str(sd["source_id"]); g = gens.get(sid)
        if g is None: continue
        canon = g["responses"][0]; cs = AR.canon_starts(canon); cm = g["cit_map"]; native = g["arm"] == "native"
        if native:
            cc = [tuple(x) for x in g["canon_cits"]]; pos_of = {}
            for k, key_ in enumerate(cc): pos_of.setdefault(key_, k)
        if (not native and len(cs) != len(cm)) or (native and len(cs) != len(cc)): continue
        s = srcs.get(sid) or srcs.get(sid.split("~s")[0]) or srcs.get(sid.split("_s")[0]); docs = docs_of(s["context"]) if s else []
        toks = sd["token_data"]; ti = 0; cache = {}
        if "~s" in sid: q_, k_ = sid.rsplit("~s", 1); key = "%s|%s" % (q_, int(k_))
        else: key = "%s|None" % sid
        for i, c in enumerate(cm):
            k = pos_of[(c["sent_idx"], c["canon_n"])] if native else i
            start = cs[k][1]; target = str(c["raw_idx"]); grp = []; acc = ""
            while ti < len(toks) and acc != target:
                acc += str(toks[ti].get("token_str", "")).strip(); grp.append(toks[ti]); ti += 1
                if len(acc) > len(target): break
            if acc != target: nfail += len(grp); continue
            if k not in cache: cache[k] = feats(canon, start, docs, tokfn)
            if cache[k] is None: nfail += len(grp); continue
            for t in grp:
                out["%s|%d" % (key, t["token_index"])] = cache[k]; nrows += 1
        nrep += 1
ev = load(a.scores); nt = sum(len([t for t in r["token_data"] if t.get("label") in (0, 1)]) for r in ev)
cov_n = sum(1 for r in ev for t in r["token_data"] if t.get("label") in (0, 1) and "%s|%s|%d" % (r["source_id"], r.get("id"), t["token_index"]) in out)
nrows = cov_n
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"))
oor = sum(1 for v in out.values() if v.get("oor") == 1); unk = sum(1 for v in out.values() if v.get("oor") is None)
print("%s: scored rows %d | with features %d (%.2f%%) | unaligned/failed %d | out-of-range rows %d | rows with unknown context %d" % (a.scores, nt, nrows, 100 * nrows / max(nt, 1), nfail, oor, unk), flush=True)
