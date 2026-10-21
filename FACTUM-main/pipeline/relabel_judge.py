"""
Replaces the gold-paragraph proxy labels of a HotpotQA Llama score file with the judge labels.
Rows (one per citation digit, in citation order) are matched to the proxy labels of hotpot_build_records.py by order
within a response; responses whose row count differs from the label count are skipped; citations dropped by the
judge (ties) are removed.

usage: relabel_judge.py <proxy response.jsonl> <in scores.jsonl> <out scores.jsonl> <judged response.jsonl> [...]
"""
import json, sys, collections
R, inp, outp = sys.argv[1:4]; judged = sys.argv[4:]
prox = {}
for l in open(R):
    r = json.loads(l); prox[r["id"]] = sorted(r["labels"], key=lambda x: x["start"])
judge = {}
for f in judged:
    for l in open(f):
        r = json.loads(l)
        for lb in r["labels"]: judge[(r["id"], lb["start"])] = 0 if lb["label_type"] == "good" else 1
out = open(outp, "w"); st = collections.Counter(); seen = set()
for l in open(inp):
    d = json.loads(l); rid = d.get("id"); rows = d["token_data"]; labs = prox.get(rid)
    if (d["source_id"], rid) in seen: continue
    seen.add((d["source_id"], rid))
    if labs is None or len(labs) != len(rows): st["skipped_responses"] += 1; continue
    new = []
    for row, lb in zip(rows, labs):
        j = judge.get((rid, lb["start"]))
        if j is None: st["dropped_ties"] += 1; continue
        row = dict(row); row["label"] = j; row["proxy_label"] = 0 if lb["label_type"] == "good" else 1; new.append(row); st["kept"] += 1; st["bad"] += j
    if new: out.write(json.dumps({"source_id": d["source_id"], "id": rid, "token_data": new}) + "\n")
print(dict(st), "judge-bad rate %.3f" % (st["bad"] / max(st["kept"], 1)))
