"""Scorer inputs from Qwen NeuCLIR generations (neuclir_qwen_generate.py output): one response record per report whose
first response contains at least one [Source: n] citation, with a placeholder label for every cited number (the
scores do not depend on the labels; the judge labels are attached afterwards with attach_qwen_labels.py), and one
source_info record per such report. Records are split round-robin into --n_chunks files
<out_dir>/{response,source_info}_<tag>_c<k>.jsonl.
"""
import argparse, json, os, re

CITE = re.compile(r'\[Source:\s*([0-9]+(?:\s*,\s*[0-9]+)*)\]')
DIGIT = re.compile(r'\d+')

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True, nargs="+", help="Generation files, read in the given order.")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--tag", required=True, help="e.g. 7b")
ap.add_argument("--n_chunks", type=int, default=4)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
resp, src = [], []
ntok = 0
for path in a.gen:
    for line in open(path):
        r = json.loads(line)
        rr = r["responses"][0]; sid = r["source_id"]; labels = []
        for m in CITE.finditer(rr):
            i0 = m.start(1)
            for d in DIGIT.finditer(m.group(1)):
                labels.append({"start": i0 + d.start(), "end": i0 + d.end(), "text": d.group(0), "meta": "none",
                               "label_type": "good", "implicit_true": "maybe", "due_to_null": True}); ntok += 1
        if not labels:
            continue
        resp.append({"id": "%s_s0" % sid, "source_id": sid, "model": r["model_tag"], "temperature": r["temperature"],
                     "labels": labels, "split": "test", "quality": "good", "response": rr})
        src.append({"source_id": sid, "prompt": r["prompt"], "context": r["context"]})
for c in range(a.n_chunks):
    rc = resp[c::a.n_chunks]; sc = src[c::a.n_chunks]
    if not rc:
        continue
    with open(os.path.join(a.out_dir, "response_%s_c%d.jsonl" % (a.tag, c)), "w") as f:
        f.write("\n".join(json.dumps(x, ensure_ascii=False) for x in rc) + "\n")
    with open(os.path.join(a.out_dir, "source_info_%s_c%d.jsonl" % (a.tag, c)), "w") as f:
        f.write("\n".join(json.dumps(x, ensure_ascii=False) for x in sc) + "\n")
print("%s: %d reports with citations, %d citation tokens -> %d chunks" % (a.tag, len(resp), ntok, a.n_chunks))
