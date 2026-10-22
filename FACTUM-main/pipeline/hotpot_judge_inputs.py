"""Judge inputs from the HotpotQA generations of the Llama models (hotpot_generate.py output): one record per question
(source_id = question id, all sampled responses), split round-robin into --n_chunks files <out_dir>/gen_<tag>_c<k>.jsonl
for judge.py.
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--gen", required=True, help="generations.jsonl of hotpot_generate.py")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--tag", required=True, help="e.g. 8b")
ap.add_argument("--n_chunks", type=int, default=4)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
out = []
for line in open(a.gen):
    g = json.loads(line)
    resps = [(x["text"] if isinstance(x, dict) and "text" in x else x["response"] if isinstance(x, dict) else x) for x in g["responses"]]
    out.append({"source_id": g["qid"], "prompt": g["prompt"], "context": g["context"], "responses": resps, "model_tag": g["model_tag"], "temperature": g["temperature"]})
for c in range(a.n_chunks):
    with open(os.path.join(a.out_dir, "gen_%s_c%d.jsonl" % (a.tag, c)), "w") as f:
        for g in out[c::a.n_chunks]:
            f.write(json.dumps(g) + "\n")
print(a.tag, "questions", len(out), "chunks", a.n_chunks)
