"""HotpotQA generation with Pleias-RAG-1B in its native prompt format
(<|query_start|>Q<|query_end|> <|source_start|><|source_id|>N title/content<|source_end|> ... <|language_start|>),
temperature 0.7, top-p 0.9, 2 samples per question (seeds 17, 18), max_new_tokens = min(--cap, 4096 - prompt length).
Records have the same schema as hotpot_gen_multi.py --arm r7bn (source_id '<qid>~s<k>', 1-based source indices).
"""
import argparse, json, os, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pleias_adapter as pa
ap = argparse.ArgumentParser()
ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model_id", default="PleIAs/Pleias-RAG-1B"); ap.add_argument("--model_tag", default="pleias-rag-1b")
ap.add_argument("--shard", default="0/1"); ap.add_argument("--max_items", type=int, default=0); ap.add_argument("--n_samples", type=int, default=2)
ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--top_p", type=float, default=0.9); ap.add_argument("--cap", type=int, default=1000); ap.add_argument("--seed", type=int, default=17); ap.add_argument("--variant", default="none", choices=["none", "suffix", "prefix", "both", "fewshot"])
a = ap.parse_args(); k, n = [int(x) for x in a.shard.split("/")]; WINDOW = 4096
def split_sources(context):
    heads = [(m.start(), int(m.group(1))) for m in re.finditer(r"(?m)^Source: (\d+)", context)]; docs, ids = [], []
    for i, (pos, nn) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(context); block = context[pos:end]
        t = re.search(r"Title:\s*(.*)", block); c = re.search(r"(?s)Content:\s*(.*)", block)
        docs.append("%s\n%s" % (t.group(1).strip() if t else "", (c.group(1).strip() if c else ""))); ids.append(nn)
    return docs, ids
INSTR = ("Answer using only the sources, in 2-4 short sentences. Each sentence must state exactly one fact (a single claim) and end with a citation of the source that directly supports it. Do not combine several facts in one sentence and do not repeat facts.")
def build_prompt(q, docs, ids):
    if a.variant == "suffix": q = q + "\n\n" + INSTR
    elif a.variant == "prefix": q = INSTR + "\n\n" + q
    elif a.variant == "both": q = INSTR + "\n\n" + q + "\n\n" + INSTR
    elif a.variant == "fewshot": q = q + "\n\n" + INSTR + " Format example: Paris is the capital of France <ref name=\"1\">quote from source 1</ref>. Paris lies on the Seine <ref name=\"2\">quote from source 2</ref>."
    return "<|query_start|>%s<|query_end|>\n" % q + "".join("<|source_start|><|source_id|>%d %s<|source_end|>\n" % (i, d) for i, d in zip(ids, docs)) + "<|language_start|>"
items = [json.loads(l) for l in open(a.items)][k::n]
if a.max_items: items = items[: a.max_items]
done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: done.add(json.loads(l)["source_id"])
        except Exception: pass
todo = [(it, s) for it in items for s in range(a.n_samples) if "%s~s%d" % (it["qid"], s) not in done]
print("[gen] pleias shard=%d/%d questions=%d samples todo=%d cap=%d" % (k, n, len(items), len(todo), a.cap), flush=True)
if not todo: print("[gen] nothing to do"); sys.exit(0)
tok = AutoTokenizer.from_pretrained(a.model_id)
model = AutoModelForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
out = open(a.out, "a", encoding="utf-8"); t0 = time.time(); EOT = tok.convert_tokens_to_ids('<|end_of_text|>'); assert EOT == 2; eos = [EOT, tok.convert_tokens_to_ids(pa.ANS1)]
for i, (it, s) in enumerate(todo, 1):
    ctx = it["context_block"]; docs, ids = split_sources(ctx); prompt = build_prompt(it["question"], docs, ids)
    enc = tok(prompt, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt").to(model.device); plen = int(enc["input_ids"].shape[1])
    mnt = min(a.cap, WINDOW - plen)
    if mnt < 200: print("[gen][SKIP] %s prompt too long (%d)" % (it["qid"], plen), flush=True); continue
    torch.manual_seed(a.seed + s)
    with torch.no_grad(): g = model.generate(**enc, do_sample=True, temperature=a.temperature, top_p=a.top_p, max_new_tokens=mnt, eos_token_id=eos, pad_token_id=EOT)
    gen_ids = g[0, plen:].tolist(); sid = "%s~s%d" % (it["qid"], s); raw = tok.decode(gen_ids, skip_special_tokens=False)
    canon, cit_map, canon_cits = pa.canonicalize_sentences(raw, ids); pr = pa.parse(raw)
    rec = {"source_id": sid, "qid": it["qid"], "sample": s, "question": it["question"], "answer": it["answer"], "type": it["type"], "level": it["level"], "num_sources": it["num_sources"],
           "gold_source_ids": it["gold_source_ids"], "num_to_title": it["num_to_title"], "task_type": "hotpotqa_distractor", "model_tag": a.model_tag, "variant": a.variant, "temperature": a.temperature, "top_p": a.top_p,
           "seed": a.seed + s, "prompt": prompt, "context": ctx, "background": None, "title": None, "gen_ids": gen_ids, "prompt_tokens": plen, "gen_tokens": len(gen_ids), "max_new_tokens": mnt,
           "arm": "native", "index_base": 1, "source_ids": ids, "response_raw": raw, "hit_max": len(gen_ids) >= mnt, "responses": [canon], "cit_map": cit_map, "canon_cits": canon_cits,
           "n_canon_cits": len(canon_cits), "n_citations": len(cit_map), "n_phantom": sum(1 for c in cit_map if c["source_n"] is None), "has_response_tag": pr["has_response_tag"],
           "answer_closed": pr["answer_closed"], "n_bare_source_id": pr["n_bare_source_id"], "plain_chars": len(pr["plain"])}
    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    if i <= 3 or i % 25 == 0: print("[gen] %d/%d %s prompt=%d gen=%d%s cites=%d answer=%s | %s" % (i, len(todo), sid, plen, len(gen_ids), "(HIT MAX)" if rec["hit_max"] else "", len(cit_map), pr["has_response_tag"], canon[:200].replace("\n", " ")), flush=True)
print("[gen] DONE", flush=True)
