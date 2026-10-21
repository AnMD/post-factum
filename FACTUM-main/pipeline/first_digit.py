"""
Collapses multi-digit citation numbers to one row: a scored token is a continuation digit if it is a digit and the
token at index - 1 is a scored digit token; only the first digit's row (scores at the point where the first digit is
generated) is kept. Checks that dropped rows carry the same label as the digit before them.

usage: first_digit.py <out_dir> name=<score file> [name=<score file> ...]   -> <out_dir>/<name>_scores.jsonl
"""
import sys, json, os
O = sys.argv[1]; os.makedirs(O, exist_ok=True)
for a in sys.argv[2:]:
    name, path = a.split("=", 1); n_in = n_out = n_lab_mismatch = n_sid_with_drop = n_cit_multi = 0
    with open(path) as fi, open("%s/%s_scores.jsonl" % (O, name), "w") as fo:
        for l in fi:
            d = json.loads(l)
            if "token_data" not in d: fo.write(l); continue
            idx = {t["token_index"]: t for t in d["token_data"]}; keep = []; dropped = 0
            for t in d["token_data"]:
                p = idx.get(t["token_index"] - 1)
                cont = p is not None and str(t["token_str"]).strip().isdigit() and str(p["token_str"]).strip().isdigit()
                n_in += 1
                if cont:
                    dropped += 1; n_cit_multi += 1
                    if p.get("label") != t.get("label"): n_lab_mismatch += 1
                else: keep.append(t); n_out += 1
            n_sid_with_drop += dropped > 0
            d["token_data"] = keep; fo.write(json.dumps(d) + "\n")
    print("%s: rows %d -> %d (dropped %d continuation-digit rows = %.1f%%); label mismatch with preceding digit: %d" % (name, n_in, n_out, n_in - n_out, 100 * (n_in - n_out) / max(n_in, 1), n_lab_mismatch), flush=True)
