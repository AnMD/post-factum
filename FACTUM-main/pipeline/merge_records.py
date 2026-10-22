"""Concatenates JSONL record files in the given order, keeping the first record of every source_id.
usage: merge_records.py <out.jsonl> <in1.jsonl> [<in2.jsonl> ...]   (missing input files are skipped)
"""
import json, sys

out, inputs = sys.argv[1], sys.argv[2:]
seen = set(); n = 0
with open(out, "w") as f:
    for fn in inputs:
        try:
            rows = [json.loads(l) for l in open(fn)]
        except FileNotFoundError:
            continue
        for d in rows:
            sid = str(d["source_id"])
            if sid in seen:
                continue
            seen.add(sid); f.write(json.dumps(d) + "\n"); n += 1
print(out, n, "records")
