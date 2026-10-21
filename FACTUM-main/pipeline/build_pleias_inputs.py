"""Builds scorer inputs from Pleias-RAG-1B generations (same contract as build_r7b_inputs.py); the context is the prompt
substring from the first <|source_start|> to the last <|source_end|>.
"""
import argparse, json
def context_block(prompt):
    a = prompt.index("<|source_start|>"); b = prompt.rindex("<|source_end|>") + len("<|source_end|>"); ctx = prompt[a:b]; assert ctx in prompt; return ctx
ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True); ap.add_argument("--out_resp", required=True); ap.add_argument("--out_src", required=True); a = ap.parse_args(); n = nl = 0
with open(a.out_resp, "w") as fr, open(a.out_src, "w") as fs:
    for line in open(a.gen):
        g = json.loads(line); raw = g["response_raw"]
        labels = [{"start": c["raw_start"], "end": c["raw_end"], "text": str(c["raw_idx"]), "meta": "none", "label_type": "good", "implicit_true": "maybe", "due_to_null": True} for c in g["cit_map"]]
        for lb in labels: assert raw[lb["start"]:lb["end"]] == lb["text"], (g["source_id"], lb)
        ctx = context_block(g["prompt"])
        fr.write(json.dumps({"id": "%s_s0" % g["source_id"], "source_id": g["source_id"], "model": g["model_tag"], "temperature": g["temperature"], "labels": labels, "split": "test", "quality": "good",
                             "response": raw, "gen_ids": g["gen_ids"]}, ensure_ascii=False) + "\n")
        fs.write(json.dumps({"source_id": g["source_id"], "task_type": "hotpot_pleias", "prompt": g["prompt"], "context": ctx, "background": None, "title": None}, ensure_ascii=False) + "\n"); n += 1; nl += len(labels)
print("[build] %d responses, %d citation-digit labels -> %s , %s" % (n, nl, a.out_resp, a.out_src))
