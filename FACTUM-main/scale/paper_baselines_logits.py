"""Per-token values for the baselines of the original FACTUM paper (run_baselines.py definitions), computed at the scored
citation tokens of a labelled score file (position t = token_index, logits = next-token logits at t - 1):
  Perplexity = exp(cross-entropy of token t); LN-Entropy = entropy of the next-token distribution;
  Energy = -logsumexp(logits); P(True) = P(True | 'Is the statement "<sentence>" true or false? Answer:'), closed book,
  where the sentence is the one containing the citation with its [Source: n] markers removed (spaCy sentences).
Also writes the log-probability and maximum probability of the token and the features of the older 9-feature
structure probe. The decoded token at token_index must equal the score file's token_str for every row of a report;
reports that cannot be aligned are written with aligned=false. Resumable; one JSON line per report.
Needs the spaCy model en_core_web_sm.
"""
import argparse, json, os, re, sys, glob, time, math
import torch
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
ap.add_argument("--model", required=True, help="Hugging Face id of the model that generated the responses.")
ap.add_argument("--family", choices=["llama", "plain"], required=True, help="llama: prompt rebuilt as the Llama scorer does (BOS + segments); plain: chat-templated prompt tokenised as is.")
ap.add_argument("--resp", nargs="+", required=True, help="Scorer response files.")
ap.add_argument("--src", nargs="+", required=True, help="Scorer source_info files.")
ap.add_argument("--scores", required=True, help="Labelled score file; its rows are the evaluated tokens.")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); C = {"model": a.model, "fam": a.family, "resp": a.resp, "src": a.src}
os.makedirs(a.out_dir, exist_ok=True); outp = "%s/shard%d.jsonl" % (a.out_dir, a.shard)
done = set()
if os.path.exists(outp):
    for l in open(outp):
        try: done.add(json.loads(l)["key"])
        except Exception: pass
print("[bl] shard", a.shard, "/", a.nshards, "already done:", len(done), flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
tok = AutoTokenizer.from_pretrained(C["model"])
model = AutoModelForCausalLM.from_pretrained(C["model"], torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()
dev = model.device
true_ids = sorted({tok.encode(t, add_special_tokens=False)[0] for t in ["True", " true", " True"]})
resp = {}; srcs = {}
for p in C["resp"]:
    for r in load(p):
        resp.setdefault(str(r["source_id"]), []).append(r)
        if r.get("id") is not None: resp.setdefault(str(r["id"]), []).append(r)
for p in C["src"]:
    for r in load(p): srcs[str(r["source_id"])] = r
print("[bl] response keys", len(resp), "source keys", len(srcs), flush=True)
recs = load(a.scores)
recs = [r for i, r in enumerate(recs) if i % a.nshards == a.shard]
if a.limit: recs = recs[:a.limit]
print("[bl] score records (this shard):", len(recs), flush=True)
def candidates(rec):
    sid = str(rec["source_id"]); rid = rec.get("id"); keys = [sid]
    if rid is not None:
        keys += [f"{sid}~s{rid}", f"{sid}_s{rid}", str(rid), str(rid).replace("_s", "~s"), f"{sid}_{rid}"]
    seen, out = set(), []
    for k in keys:
        for r in resp.get(k, []):
            if id(r) not in seen: seen.add(id(r)); out.append(r)
    return out
def prefix_ids(fam, prompt, context):
    if fam == "llama":
        bos = torch.tensor([[tok.bos_token_id]], dtype=torch.long)
        pb, pa = prompt.split(context, 1)
        d = "\n\n**Instructions:**"
        if d in pa: q, ins = pa.split(d, 1); ins = d + ins
        else: q, ins = pa, ""
        enc = lambda t: tok(t, add_special_tokens=False, return_tensors="pt").input_ids
        return torch.cat([bos, enc(pb), enc(context), enc(q), enc(ins)], dim=1)
    return tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
def build(rec_in, fam):
    s = srcs.get(str(rec_in["source_id"]))
    if s is None: return None
    pre = prefix_ids(fam, s["prompt"], s["context"]); rid = tok(rec_in["response"], add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True)
    return pre, rid.input_ids, rid["offset_mapping"][0].tolist()
@torch.no_grad()
def next_logits(ids, pos):
    h = model.model(input_ids=ids.to(dev), use_cache=False).last_hidden_state[0, pos]
    lg = model.lm_head(h)
    sc = getattr(model.config, "logit_scale", None)
    if sc: lg = lg * sc
    return lg.float()
@torch.no_grad()
def ptrue(stmt):
    q = tok(f'Is the statement "{stmt}" true or false? Answer:', return_tensors="pt").input_ids.to(dev)
    out = next_logits(q, [q.shape[1] - 1])[0]
    return torch.softmax(out, -1)[true_ids].sum().item()
CITE = re.compile(r"\[\s*(?:Source|Sources)s?\s*:?[^\]]*\]\.?", re.I)
def stmt_for(resp_text, char_start, sents):
    for i, s in enumerate(sents):
        if s.start_char <= char_start < s.end_char:
            c = CITE.sub("", s.text).strip()
            if not c and i > 0: c = sents[i - 1].text.strip()
            return c
    return ""

BR = re.compile(r"\[[^\]]*?Source[^\]]*\]", re.I)
def struct_feats(text, cs, sent):
    """text-only citation-structure features (no model internals): used by the structure-only baseline."""
    brs = [(m.start(), m.end(), [int(x) for x in re.findall(r"\d+", m.group(0))]) for m in BR.finditer(text)]
    nprev = 0; seen = set(); cited = -1; kin = 0; nb = 1; first = 1
    for b0, b1, nums in brs:
        if b0 <= cs < b1:
            rel = cs - b0; off = 0; kin = 0
            for q, mm in enumerate(re.finditer(r"\d+", text[b0:b1])):
                if mm.start() <= rel < mm.end(): cited = int(mm.group(0)); kin = q; break
            nb = len(nums); first = int(cited not in seen); break
        nprev += len(nums); seen.update(nums)
    ncs = 0
    if sent is not None:
        ncs = sum(len(nums) for b0, b1, nums in brs if sent.start_char <= b0 < sent.end_char)
    return {"rel_pos": cs / max(len(text), 1), "resp_chars": len(text), "cited_num": cited, "k_in_bracket": kin, "n_in_bracket": nb, "n_prev_cites": nprev, "first_of_source": first, "n_cites_in_sent": ncs}
fo = open(outp, "a"); n_ok = n_bad = 0; t0 = time.time()
for ri, rec in enumerate(recs):
    key = "%s|%s" % (rec["source_id"], rec.get("id"))
    if key in done: continue
    rows = [t for t in rec["token_data"] if t.get("label") in (0, 1)]
    if not rows: continue
    best = None
    for cand in candidates(rec):
        b = build(cand, C["fam"])
        if b is None: continue
        pre, rid, offs = b; full = torch.cat([pre, rid], dim=1)
        bad = sum(1 for t in rows if t["token_index"] >= full.shape[1] or tok.decode(full[0, t["token_index"]]).strip() != str(t["token_str"]).strip())
        if best is None or bad < best[0]: best = (bad, cand, pre, rid, offs, full)
        if bad == 0: break
    if best is None:
        fo.write(json.dumps({"key": key, "aligned": False, "reason": "no candidate", "n_rows": len(rows)}) + "\n"); fo.flush(); n_bad += 1; continue
    bad, cand, pre, rid, offs, full = best
    if bad:
        fo.write(json.dumps({"key": key, "aligned": False, "reason": "mismatch %d/%d" % (bad, len(rows)), "n_rows": len(rows)}) + "\n"); fo.flush(); n_bad += 1; continue
    pos = [t["token_index"] - 1 for t in rows]
    lg = next_logits(full, pos)
    tgt = torch.tensor([t["token_index"] for t in rows], device=lg.device); tgt_ids = full[0, [t["token_index"] for t in rows]].to(lg.device)
    logp = torch.log_softmax(lg, -1); p = logp.exp()
    lp_t = logp[torch.arange(len(rows)), tgt_ids]; ent = -(p * logp).sum(-1); energy = -torch.logsumexp(lg, -1); maxp = p.max(-1).values
    sents = list(nlp(cand["response"]).sents); plen = pre.shape[1]; res_rows = []
    for j, t in enumerate(rows):
        ci = t["token_index"] - plen
        cs = offs[ci][0] if 0 <= ci < len(offs) else 0
        st = stmt_for(cand["response"], cs, sents)
        sent_obj = next((x for x in sents if x.start_char <= cs < x.end_char), None)
        row = {"token_index": t["token_index"], "label": t["label"], "ppl": float(torch.exp(-lp_t[j])), "ln_entropy": float(ent[j]), "energy": float(energy[j]),
               "logp_tok": float(lp_t[j]), "maxprob": float(maxp[j]), "stmt": st, "ptrue": None}
        row.update(struct_feats(cand["response"], cs, sent_obj)); row["claim_words"] = len(st.split())
        res_rows.append(row)
    cache = {}
    for r in res_rows:
        if r["stmt"]:
            if r["stmt"] not in cache: cache[r["stmt"]] = ptrue(r["stmt"])
            r["ptrue"] = cache[r["stmt"]]
        r.pop("stmt")
    fo.write(json.dumps({"key": key, "aligned": True, "cand_id": cand.get("id"), "rows": res_rows}) + "\n"); fo.flush(); n_ok += 1
    if (n_ok + n_bad) % 25 == 0: print("[bl] %d/%d reports ok=%d bad=%d  %.1f min" % (ri + 1, len(recs), n_ok, n_bad, (time.time() - t0) / 60), flush=True)
print("[bl] DONE ok=%d unaligned=%d  %.1f min" % (n_ok, n_bad, (time.time() - t0) / 60), flush=True)
