"""Builds scorer inputs from Command R7B generations: a response file with one placeholder label per raw citation digit
(raw character coordinates, the scored text is the raw output) and a source_info file whose context is the exact
prompt substring that holds the documents. Scores do not depend on the labels; the judge labels are attached
afterwards with attach_r7b.py.
"""
import argparse, json, re

def native_context_block(prompt):
    """The documents-as-tool-result JSON block inside the rendered rag prompt (docs are JSON-escaped, so NOT verbatim `context`)."""
    a = prompt.index('"results": {') + len('"results": {')
    b = prompt.index('        },\n        "is_error"', a)
    ctx = prompt[a:b].strip("\n")
    assert ctx.strip().startswith('"0":'), ctx[:60]
    assert ctx in prompt
    return ctx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True); ap.add_argument("--out_resp", required=True); ap.add_argument("--out_src", required=True)
    a = ap.parse_args()
    n = nl = 0
    with open(a.out_resp, "w") as fr, open(a.out_src, "w") as fs:
        for line in open(a.gen):
            g = json.loads(line)
            labels = [{"start": c["raw_start"], "end": c["raw_end"], "text": str(c["raw_idx"]), "meta": "none",
                       "label_type": "good", "implicit_true": "maybe", "due_to_null": True} for c in g["cit_map"]]
            raw = g["response_raw"]
            for lb in labels:
                assert raw[lb["start"]:lb["end"]] == lb["text"], (g["source_id"], lb)
            ctx = native_context_block(g["prompt"]) if g["arm"] == "native" else g["context"]
            assert ctx in g["prompt"], "context not verbatim in prompt for sid %s" % g["source_id"]
            fr.write(json.dumps({"id": "%s_s0" % g["source_id"], "source_id": g["source_id"], "model": g["model_tag"],
                                 "temperature": g["temperature"], "labels": labels, "split": "test", "quality": "good",
                                 "response": raw, "gen_ids": g["gen_ids"]}, ensure_ascii=False) + "\n")
            fs.write(json.dumps({"source_id": g["source_id"], "task_type": "neuclir_r7b", "prompt": g["prompt"], "context": ctx,
                                 "background": g.get("background"), "title": g.get("title")}, ensure_ascii=False) + "\n")
            n += 1; nl += len(labels)
    print("[build] %d reports, %d citation-digit labels -> %s , %s" % (n, nl, a.out_resp, a.out_src))

if __name__ == "__main__":
    main()
