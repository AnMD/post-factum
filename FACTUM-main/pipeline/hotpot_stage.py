"""
Stage helpers for the HotpotQA generations of Qwen, Command R7B and Pleias.

  hotpot_stage.py merge <base_dir> <name>
      merges <base_dir>/gen/gen_<name>_s*.jsonl (deduplicated by source_id) into <base_dir>/gen_<name>.jsonl and
      writes four judge chunks <base_dir>/judge_in/gen_<name>_c{0..3}.jsonl (both samples of a question in the same chunk).
  hotpot_stage.py finalize <name> <in.jsonl> <out.jsonl>
      restores the question ids of a labelled score file: source_id = question id, id = sample index; keeps rows
      with label 0/1 and prints the number of responses, citations and the hallucination rate.
"""
import sys, json, glob, os, collections
cmd = sys.argv[1]
if cmd == "merge":
    B, name = sys.argv[2], sys.argv[3]; seen = {}
    for f in sorted(glob.glob("%s/gen/gen_%s_s*.jsonl" % (B, name))):
        for l in open(f):
            try: d = json.loads(l)
            except Exception: continue
            seen[d["source_id"]] = d
    recs = sorted(seen.values(), key=lambda d: (d["qid"], d["sample"])); os.makedirs(B + "/judge_in", exist_ok=True)
    with open("%s/gen_%s.jsonl" % (B, name), "w") as fo:
        for d in recs: fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    qs = sorted({d["qid"] for d in recs}); qc = {q: i % 4 for i, q in enumerate(qs)}; fs = [open("%s/judge_in/gen_%s_c%d.jsonl" % (B, name, c), "w") for c in range(4)]
    for d in recs: fs[qc[d["qid"]]].write(json.dumps(d, ensure_ascii=False) + "\n")
    print(name, "records", len(recs), "questions", len(qs), "per chunk", collections.Counter(qc[d["qid"]] for d in recs))
elif cmd == "finalize":
    name, inp, outp = sys.argv[2:5]; seen = set(); n = nb = 0; fo = open(outp, "w")
    for l in open(inp):
        try: d = json.loads(l)
        except Exception: continue
        sid = str(d["source_id"])
        if "~s" not in sid or sid in seen: continue
        seen.add(sid); qid, k = sid.rsplit("~s", 1); rows = [t for t in d["token_data"] if t.get("label") in (0, 1)]
        if not rows: continue
        fo.write(json.dumps({"source_id": qid, "id": int(k), "token_data": rows}) + "\n"); n += len(rows); nb += sum(t["label"] for t in rows)
    print(name, "responses", len(seen), "citations", n, "hallucinated %.1f%%" % (100 * nb / max(n, 1)))
