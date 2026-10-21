"""Builds the scorer's response.jsonl / source_info.jsonl from hotpot_generate.py output: two response records per
question (samples s0, s1; both with source_id = question id) and one source record per question. Each citation digit
gets a character-span label; good if the cited paragraph is a gold supporting paragraph (proxy label, replaced by the
judge label later with relabel_judge.py). The leading BOS string of the stored prompt is stripped, because the scorer
prepends one BOS itself. Prints label and contrast statistics.
"""
import argparse
import collections
import json
import re

CITE_RE = re.compile(r"\[Source:\s*([0-9]+(?:\s*,\s*[0-9]+)*)\]")
DIGIT_RE = re.compile(r"\d+")


def parse_citations(resp, gold_ids, num_sources):
    """Return (labels, stats) where labels land char spans on each cited digit in `resp`."""
    labels = []
    n_multi = 0
    n_fab = 0
    for m in CITE_RE.finditer(resp):
        inner = m.group(1)
        inner_start = m.start(1)
        digits = list(DIGIT_RE.finditer(inner))
        if len(digits) > 1:
            n_multi += 1
        for d in digits:
            n = int(d.group(0))
            start = inner_start + d.start()
            end = inner_start + d.end()
            fabricated = not (1 <= n <= num_sources)
            if fabricated:
                n_fab += 1
            is_good = (n in gold_ids) and not fabricated
            labels.append({
                "start": start,
                "end": end,
                "text": str(n),
                "meta": "none",
                "label_type": "good" if is_good else "bad",
                "implicit_true": "maybe",
                "due_to_null": True,
            })
    return labels, {"multi": n_multi, "fab": n_fab}


def config():
    p = argparse.ArgumentParser(description="P2 build FACTUM records + gate stats")
    p.add_argument("--gen", required=True, help="generations jsonl from hotpot_generate.py")
    p.add_argument("--out_response", required=True)
    p.add_argument("--out_source", required=True)
    p.add_argument("--strip_prompt_prefix", default="<|begin_of_text|>",
                   help="leading token string to strip from stored prompt (Llama-3 BOS). '' disables.")
    return p.parse_args()


def main():
    args = config()
    gens = [json.loads(l) for l in open(args.gen, "r", encoding="utf-8")]

    resp_records = []
    src_records = []
    seen_src = set()
    n_stripped = 0

    tot_cit = good_cit = bad_cit = fab_cit = multi_cit = 0
    cits_per_record = []
    rec_survives = []
    q_has_survivor = collections.defaultdict(bool)
    n_records = 0
    n_no_citation = 0

    for g in gens:
        qid = g["qid"]
        gold_ids = set(g["gold_source_ids"])
        nsrc = g["num_sources"]

        if qid not in seen_src:
            seen_src.add(qid)
            prompt = g["prompt"]
            if args.strip_prompt_prefix and prompt.startswith(args.strip_prompt_prefix):
                prompt = prompt[len(args.strip_prompt_prefix):]
                n_stripped += 1
            src_records.append({
                "source_id": qid,
                "task_type": "hotpotqa_distractor",
                "prompt": prompt,
                "context": g["context"],
                "question": g["question"],
                "answer": g["answer"],
                "type": g["type"],
                "level": g["level"],
                "num_sources": nsrc,
                "gold_source_ids": sorted(gold_ids),
                "num_to_title": g["num_to_title"],
            })

        for s, resp in enumerate(g["responses"]):
            n_records += 1
            labels, st = parse_citations(resp, gold_ids, nsrc)
            ncit = len(labels)
            cits_per_record.append(ncit)
            if ncit == 0:
                n_no_citation += 1
            g_ct = sum(1 for l in labels if l["label_type"] == "good")
            b_ct = ncit - g_ct
            tot_cit += ncit
            good_cit += g_ct
            bad_cit += b_ct
            fab_cit += st["fab"]
            multi_cit += st["multi"]
            survives = (g_ct > 0 and b_ct > 0)
            rec_survives.append(survives)
            if survives:
                q_has_survivor[qid] = True

            resp_records.append({
                "id": f"{qid}_s{s}",
                "source_id": qid,
                "model": g["model_tag"],
                "temperature": g["temperature"],
                "labels": labels,
                "split": "test",
                "quality": "good",
                "response": resp,
            })

    with open(args.out_response, "w", encoding="utf-8") as f:
        for r in resp_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out_source, "w", encoding="utf-8") as f:
        for r in src_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_q = len(seen_src)
    n_surv_rec = sum(rec_survives)
    n_surv_q = sum(1 for q in seen_src if q_has_survivor[q])
    dist = collections.Counter(cits_per_record)
    print("=" * 72)
    print("PILOT-GATE STATS (HotpotQA record build)")
    print("=" * 72)
    print(f"questions                 : {n_q}")
    print(f"response records (2/q)    : {n_records}")
    print(f"records with 0 citations  : {n_no_citation}")
    print(f"total citations parsed    : {tot_cit}")
    print(f"  good (cited gold)       : {good_cit}")
    print(f"  bad  (hallucinated)     : {bad_cit}   -> HALLUCINATION RATE = {bad_cit/max(tot_cit,1):.3f}"
          f"   (NeuCLIR base 0.187)")
    print(f"  fabricated-index cits   : {fab_cit}   ({fab_cit/max(tot_cit,1):.3f} of all)")
    print(f"  multi-source cite groups: {multi_cit}")
    print(f"citations/record dist     : {dict(sorted(dist.items()))}  (mean {tot_cit/max(n_records,1):.2f})")
    print("-" * 72)
    print(f"CONTRAST FILTER (extractor drops records with all-good or all-bad):")
    print(f"  records that SURVIVE     : {n_surv_rec}/{n_records}  ({n_surv_rec/max(n_records,1):.1%})")
    print(f"  questions w/ >=1 survivor: {n_surv_q}/{n_q}  ({n_surv_q/max(n_q,1):.1%})  <- usable cluster count")
    print("=" * 72)
    print(f"[build] wrote {len(resp_records)} response records -> {args.out_response}")
    print(f"[build] wrote {len(src_records)} source records  -> {args.out_source}"
          f"  (stripped leading BOS from {n_stripped}/{len(src_records)} prompts)")
    if n_surv_q < 0.5 * n_q:
        print("[build][GATE] <50% of questions survive the contrast filter -> consider the "
              "gold-absent forcing lever before the 500-main run.")


if __name__ == "__main__":
    main()
