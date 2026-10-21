"""HotpotQA generation with Qwen2.5-Instruct (--arm qwen) and Command R7B with native grounded citations (--arm r7bn):
the same questions and prompt as hotpot_generate.py, temperature 0.7, top-p 0.9, 2 samples per question (seeds 17, 18),
max_new_tokens 320. Every sample is its own record with source_id '<qid>~s<k>'. For R7B the sources are passed as
documents and the native citations are canonicalised with cite_adapter. Resumable (appends one record per sample).
"""
import argparse, json, os, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hotpot_generate import build_user_message
import cite_adapter as ca
ap = argparse.ArgumentParser()
ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--arm", required=True, choices=["qwen", "r7bn"]); ap.add_argument("--model_id", required=True); ap.add_argument("--model_tag", required=True)
ap.add_argument("--shard", default="0/1"); ap.add_argument("--max_items", type=int, default=0)
ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--top_p", type=float, default=0.9); ap.add_argument("--n_samples", type=int, default=2)
ap.add_argument("--max_new_tokens", type=int, default=320); ap.add_argument("--seed", type=int, default=17)
a = ap.parse_args(); k, n = [int(x) for x in a.shard.split("/")]
CITE_BULLET = re.compile(r"\n- Add a citation of the form \[Source: #\][^\n]*")
def split_sources(context):
    heads = [(m.start(), int(m.group(1))) for m in re.finditer(r"(?m)^Source: (\d+)", context)]; docs, ids = [], []
    for i, (pos, nn) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(context); block = context[pos:end]
        t = re.search(r"Title:\s*(.*)", block); c = re.search(r"(?s)Content:\s*(.*)", block)
        docs.append({"heading": "Source %d: %s" % (nn, t.group(1).strip() if t else ""), "body": (c.group(1).strip() if c else "")}); ids.append(nn)
    return docs, ids
def rep1(t, old, new):
    assert t.count(old) == 1, "expected exactly one %r, found %d" % (old[:50], t.count(old)); return t.replace(old, new, 1)
def native_user_text(ut, context):
    block = "---\n**Context:**\n" + context + "\n---\n"; ut = rep1(ut, block, "---\n")
    ut = rep1(ut, "the Context below", "the provided documents"); ut = rep1(ut, "mentioned in the Context", "mentioned in the documents")
    ut, nsub = CITE_BULLET.subn("", ut); assert nsub == 1, "citation bullet not found exactly once"; return ut
items = [json.loads(l) for l in open(a.items)][k::n]
if a.max_items: items = items[: a.max_items]
done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: done.add(json.loads(l)["source_id"])
        except Exception: pass
todo = [(it, s) for it in items for s in range(a.n_samples) if "%s~s%d" % (it["qid"], s) not in done]
print("[gen] arm=%s shard=%d/%d questions=%d samples todo=%d" % (a.arm, k, n, len(items), len(todo)), flush=True)
if not todo: print("[gen] nothing to do"); sys.exit(0)
tok = AutoTokenizer.from_pretrained(a.model_id)
model = AutoModelForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.bfloat16, device_map="auto", **({"attn_implementation": "sdpa"} if a.arm == "r7bn" else {})).eval()
out = open(a.out, "a", encoding="utf-8"); t0 = time.time()
for i, (it, s) in enumerate(todo, 1):
    ctx = it["context_block"]; ut = build_user_message(it["question"], ctx)
    if a.arm == "qwen":
        prompt = tok.apply_chat_template([{"role": "user", "content": ut}], tokenize=False, add_generation_prompt=True); docs_ids = None
        if prompt.count(ctx) != 1: print("[gen][SKIP] %s context not unique" % it["qid"]); continue
    else:
        docs, docs_ids = split_sources(ctx)
        prompt = tok.apply_chat_template([{"role": "user", "content": native_user_text(ut, ctx)}], documents=docs, tokenize=False, add_generation_prompt=True, enable_citations=True)
    enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device); torch.manual_seed(a.seed + s)
    gkw = dict(do_sample=True, temperature=a.temperature, top_p=a.top_p, max_new_tokens=a.max_new_tokens)
    with torch.no_grad(): g = model.generate(**enc, **gkw, **({"disable_compile": True} if a.arm == "r7bn" else {"pad_token_id": tok.eos_token_id}))
    gen_ids = g[0, enc["input_ids"].shape[1]:].tolist(); sid = "%s~s%d" % (it["qid"], s)
    base = {"source_id": sid, "qid": it["qid"], "sample": s, "question": it["question"], "answer": it["answer"], "type": it["type"], "level": it["level"], "num_sources": it["num_sources"],
            "gold_source_ids": it["gold_source_ids"], "num_to_title": it["num_to_title"], "task_type": "hotpotqa_distractor", "model_tag": a.model_tag, "temperature": a.temperature, "top_p": a.top_p,
            "seed": a.seed + s, "prompt": prompt, "context": ctx, "background": None, "title": None, "gen_ids": gen_ids, "prompt_tokens": int(enc["input_ids"].shape[1]), "gen_tokens": len(gen_ids)}
    if a.arm == "qwen":
        rec = dict(base, responses=[tok.decode(gen_ids, skip_special_tokens=True).strip()])
    else:
        raw = tok.decode(gen_ids, skip_special_tokens=False); canon, cit_map, canon_cits = ca.canonicalize_sentences(raw, 0, docs_ids); pr = ca.parse(raw)
        rec = dict(base, arm="native", index_base=0, source_ids=docs_ids, response_raw=raw, hit_max=len(gen_ids) >= a.max_new_tokens, responses=[canon], cit_map=cit_map, canon_cits=canon_cits,
                   n_canon_cits=len(canon_cits) if canon_cits is not None else len(cit_map), n_citations=len(cit_map), n_phantom=sum(1 for c in cit_map if c["source_n"] is None),
                   has_response_tag=pr["has_response_tag"], plain_chars=len(pr["plain"]))
    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    if i <= 2 or i % 25 == 0: print("[gen] %d/%d %s gen_tokens=%d elapsed=%.0fs | %s" % (i, len(todo), sid, len(gen_ids), time.time() - t0, rec["responses"][0][:160].replace("\n", " ")), flush=True)
print("[gen] DONE", flush=True)
