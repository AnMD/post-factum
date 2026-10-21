"""HotpotQA data preparation. Reads the official distractor dev set (hotpot_dev_distractor_v1.json, or a JSONL export
of it) and writes one record per question with the context block ("Source: N / Title / Content", the 10 paragraphs
shuffled with a per-question seed) and the source numbers of the gold paragraphs. A seeded shuffle of the questions
gives disjoint pilot and main sets.
"""
import argparse
import hashlib
import json
import random
import collections


def config():
    p = argparse.ArgumentParser(description="P0 HotpotQA prep")
    p.add_argument("--hotpot_json", required=True, help="Path to hotpot_dev_distractor_v1.json")
    p.add_argument("--out_dir", required=True, help="Directory to write hotpot_pilot.jsonl / hotpot_main.jsonl")
    p.add_argument("--seed", type=int, default=17, help="Global seed (matches FACTUM analysis seed=17)")
    p.add_argument("--n_pilot", type=int, default=20)
    p.add_argument("--n_main", type=int, default=500)
    p.add_argument("--types", default="bridge,comparison", help="comma-sep question types to keep")
    p.add_argument("--levels", default="", help="optional comma-sep level filter (blank = all)")
    return p.parse_args()


def item_rng(seed, qid):
    """Deterministic per-item RNG (independent of PYTHONHASHSEED)."""
    h = hashlib.md5(f"{seed}:{qid}".encode("utf-8")).hexdigest()
    return random.Random(int(h, 16))


def load_records(path):
    """Accept either the original HotpotQA JSON *array* or a JSONL file (one record per line,
    e.g. HuggingFace `datasets` .to_json output). Discriminate by the first non-space char."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def normalize_record(rec):
    """Return a common internal form regardless of source schema.
      * original:  _id, context=[[title,[sents]],...], supporting_facts=[[title,sent_id],...]
      * HF hub  :  id,  context={"title":[...],"sentences":[[...]]},
                   supporting_facts={"title":[...],"sent_id":[...]}
    -> {qid, question, answer, type, level, paras=[(title,[sents]),...], gold_titles=set}
    """
    qid = rec.get("_id") or rec.get("id")
    ctx = rec["context"]
    if isinstance(ctx, dict):
        paras = [(t, list(s)) for t, s in zip(ctx["title"], ctx["sentences"])]
    else:
        paras = [(t, list(s)) for (t, s) in ctx]
    sf = rec["supporting_facts"]
    if isinstance(sf, dict):
        gold_titles = set(sf["title"])
    else:
        gold_titles = {row[0] for row in sf}
    return {
        "qid": qid,
        "question": rec["question"],
        "answer": rec["answer"],
        "type": rec.get("type", ""),
        "level": rec.get("level", ""),
        "paras": paras,
        "gold_titles": gold_titles,
    }


def build_context_block(shuffled_paras):
    """shuffled_paras: list of (title, [sentences]) already permuted. Number Source: 1..K in order.
    Returns (context_block_str, {source_num: title})."""
    parts = []
    num_to_title = {}
    for n, (title, sents) in enumerate(shuffled_paras, start=1):
        content = "".join(sents).strip()
        num_to_title[n] = title
        parts.append(f"Source: {n}\nTitle: {title}\nContent: {content}")
    return "\n\n".join(parts), num_to_title


def prep_item(nrec, seed):
    qid = nrec["qid"]
    paras = nrec["paras"]
    gold_titles = nrec["gold_titles"]

    ctx_titles = {t for (t, _) in paras}
    missing_gold = sorted(gold_titles - ctx_titles)

    shuffled = paras[:]
    item_rng(seed, qid).shuffle(shuffled)
    context_block, num_to_title = build_context_block(shuffled)
    gold_source_ids = sorted(n for n, t in num_to_title.items() if t in gold_titles)

    return {
        "qid": qid,
        "question": nrec["question"],
        "answer": nrec["answer"],
        "type": nrec["type"],
        "level": nrec["level"],
        "num_sources": len(shuffled),
        "context_block": context_block,
        "num_to_title": {str(k): v for k, v in num_to_title.items()},
        "gold_source_ids": gold_source_ids,
        "gold_titles": sorted(gold_titles),
        "missing_gold_titles": missing_gold,
    }


def main():
    args = config()
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    keep_types = {t.strip() for t in args.types.split(",") if t.strip()}
    keep_levels = {l.strip() for l in args.levels.split(",") if l.strip()}

    data = load_records(args.hotpot_json)
    print(f"[prep] loaded {len(data)} HotpotQA distractor-dev records from {args.hotpot_json}")
    data = [normalize_record(rec) for rec in data]

    pool = []
    for rec in data:
        if keep_types and rec["type"] not in keep_types:
            continue
        if keep_levels and rec["level"] not in keep_levels:
            continue
        pool.append(rec)
    print(f"[prep] after type={sorted(keep_types)} level={sorted(keep_levels) or 'ALL'}: {len(pool)} records")
    print(f"[prep] type dist: {dict(collections.Counter(r.get('type') for r in pool))}")
    print(f"[prep] level dist: {dict(collections.Counter(r.get('level') for r in pool))}")

    order = list(range(len(pool)))
    random.Random(args.seed).shuffle(order)
    need = args.n_pilot + args.n_main
    if len(order) < need:
        print(f"[prep][WARN] pool ({len(order)}) < requested pilot+main ({need}); taking what exists")
    pilot_idx = order[: args.n_pilot]
    main_idx = order[args.n_pilot: args.n_pilot + args.n_main]

    def write_split(name, idxs):
        path = os.path.join(args.out_dir, name)
        n_missing = 0
        with open(path, "w", encoding="utf-8") as out:
            for i in idxs:
                item = prep_item(pool[i], args.seed)
                if item["missing_gold_titles"]:
                    n_missing += 1
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[prep] wrote {len(idxs)} -> {path}  (items with a missing gold title: {n_missing})")

    write_split("hotpot_pilot.jsonl", pilot_idx)
    write_split("hotpot_main.jsonl", main_idx)
    print("[prep] done.")


if __name__ == "__main__":
    main()
