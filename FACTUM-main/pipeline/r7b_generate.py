"""NeuCLIR report generation with Command R7B (temperature 0.7, top-p 0.9, max_new_tokens 1400, seed 17, bf16).
--arm qwenfmt: the FACTUM user prompt unchanged (sources pasted in the message, [Source: #] instruction).
--arm native: grounded generation; the sources are passed as documents ({heading: 'Source N: Title', body}) and the
model cites natively (0-based positions in the document list). The task text is FACTUM's, with the pasted context
block and the [Source: #] instruction removed and 'the Context' replaced by 'the provided documents'.
--source_info expects the Qwen-templated source_info of neuclir_qwen_generate.py / judge.py. Resumable.
"""
import argparse, json, os, re, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cite_adapter as ca

CITE_BULLET = re.compile(r'- End \*\*every sentence\*\* with a citation of the form \[Source: #\][^\n]*\n')
MODEL_ID = "CohereLabs/c4ai-command-r7b-12-2024"

def user_text_from_prompt(p):
    s = p.index("<|im_start|>user\n") + len("<|im_start|>user\n"); e = p.index("<|im_end|>", s)
    return p[s:e]

def split_sources(context):
    heads = [(m.start(), int(m.group(1))) for m in re.finditer(r"(?m)^Source: (\d+)", context)]
    docs, ids = [], []
    for i, (pos, n) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(context)
        block = context[pos:end]
        t = re.search(r"Title:\s*(.*)", block); c = re.search(r"(?s)Content:\s*(.*)", block)
        docs.append({"heading": "Source %d: %s" % (n, t.group(1).strip() if t else ""), "body": (c.group(1).strip() if c else "")})
        ids.append(n)
    return docs, ids

def replace_once(text, old, new):
    assert text.count(old) == 1, "expected exactly one occurrence of %r, found %d" % (old[:60], text.count(old))
    return text.replace(old, new, 1)

def native_user_text(ut, context):
    block = "---\n**Context:**\n" + context + "\n---\n"
    assert ut.count(block) == 1, "context block not found exactly once in user text"
    ut = ut.replace(block, "---\n", 1)
    ut = replace_once(ut, "the Context below", "the provided documents")
    ut = replace_once(ut, "mentioned in the Context", "mentioned in the documents")
    ut, n = CITE_BULLET.subn("", ut); assert n == 1, "citation bullet not found exactly once"
    return ut

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_info", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["native", "qwenfmt"], required=True)
    ap.add_argument("--ids", default="", help="comma-separated source_ids (small tests)")
    ap.add_argument("--shard", default="0/1", help="k/n round-robin shard")
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=1400); ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    k, n = [int(x) for x in a.shard.split("/")]

    recs = [json.loads(l) for l in open(a.source_info)]
    if a.ids: want = set(a.ids.split(",")); recs = [r for r in recs if str(r["source_id"]) in want]
    recs = recs[k::n]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(str(json.loads(l)["source_id"]))
            except Exception: pass
    todo = [r for r in recs if str(r["source_id"]) not in done]
    print("[gen] arm=%s shard=%d/%d records=%d already_done=%d todo=%d" % (a.arm, k, n, len(recs), len(done), len(todo)), flush=True)
    if not todo: print("[gen] nothing to do"); return

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
    tag = "command-r7b-12-2024-" + a.arm
    out = open(a.out, "a", encoding="utf-8")
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        sid = str(r["source_id"]); context = r["context"]
        ut = user_text_from_prompt(r["prompt"]); docs, source_ids = split_sources(context)
        if a.arm == "qwenfmt":
            prompt = tok.apply_chat_template([{"role": "user", "content": ut}], tokenize=False, add_generation_prompt=True)
            index_base = 1
        else:
            prompt = tok.apply_chat_template([{"role": "user", "content": native_user_text(ut, context)}], documents=docs,
                                             tokenize=False, add_generation_prompt=True, enable_citations=True)
            index_base = 0
        enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        torch.manual_seed(a.seed)
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=a.temperature, top_p=a.top_p,
                               max_new_tokens=a.max_new_tokens, disable_compile=True)
        gen_ids = g[0, enc["input_ids"].shape[1]:].tolist()
        raw = tok.decode(gen_ids, skip_special_tokens=False)
        if a.arm == "native":
            canon, cit_map, canon_cits = ca.canonicalize_sentences(raw, index_base, source_ids)
        else:
            canon, cit_map = ca.canonicalize(raw, index_base, source_ids); canon_cits = None
        n_ph = sum(1 for c in cit_map if c["source_n"] is None)
        pr = ca.parse(raw)
        rec = {"source_id": r["source_id"], "model_tag": tag, "arm": a.arm, "temperature": a.temperature, "top_p": a.top_p, "seed": a.seed,
               "prompt": prompt, "context": context, "background": r.get("background"), "title": r.get("title"),
               "source_ids": source_ids, "index_base": index_base,
               "response_raw": raw, "gen_ids": gen_ids, "prompt_tokens": int(enc["input_ids"].shape[1]), "gen_tokens": len(gen_ids),
               "hit_max": len(gen_ids) >= a.max_new_tokens, "responses": [canon], "cit_map": cit_map, "canon_cits": canon_cits,
               "n_canon_cits": (len(canon_cits) if canon_cits is not None else len(cit_map)),
               "n_citations": len(cit_map), "n_phantom": n_ph, "has_response_tag": pr["has_response_tag"],
               "n_thinking_blocks": raw.count("<|START_THINKING|>"), "n_action_blocks": raw.count("<|START_ACTION|>"),
               "plain_chars": len(pr["plain"])}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        el = time.time() - t0; eta = el / i * (len(todo) - i)
        print("[gen] %d/%d sid=%s prompt=%d gen=%d%s raw_cites=%d judged_cites=%d phantom=%d resp_tag=%s elapsed=%.0fs eta=%.0fs" % (
            i, len(todo), sid, rec["prompt_tokens"], rec["gen_tokens"], "(HIT MAX)" if rec["hit_max"] else "", rec["n_citations"],
            rec["n_canon_cits"], n_ph, rec["has_response_tag"], el, eta), flush=True)
    print("[gen] DONE", flush=True)

if __name__ == "__main__":
    main()
