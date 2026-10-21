"""Pleias-RAG-1B output adapter. The model reasons first and then answers inside <|answer_start|>..<|answer_end|>,
citing with <ref name="<|source_id|>N">quote</ref> (or name="N"). Only citations inside the answer count; the cited
digits N (1-based source numbers) are the scored tokens. parse() has the same contract as cite_adapter.parse(), so
cite_adapter.canonicalize_sentences() is reused. Run this file to execute its self-tests.
"""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cite_adapter as ca

ANS0, ANS1 = "<|answer_start|>", "<|answer_end|>"
REF = re.compile(r'<ref name="(?:<\|source_id\|>)?(\d+)">(.*?)</ref>', re.S)
SPECIAL = re.compile(r"<\|[a-z_]+\|>|<\|begin_of_text\|>|<\|end_of_text\|>")

def parse(raw):
    a0 = raw.find(ANS0); has_tag = a0 >= 0
    rs = a0 + len(ANS0) if has_tag else len(raw)
    a1 = raw.find(ANS1, rs); re_end = a1 if a1 >= 0 else len(raw)
    plain_parts, cits = [], []; plain_len = 0; cursor = rs; prev_cit_plain = 0; group = 0
    def add(seg):
        nonlocal plain_len
        seg = SPECIAL.sub("", seg); plain_parts.append(seg); plain_len += len(seg)
    for m in REF.finditer(raw, rs, re_end):
        add(raw[cursor:m.start()]); cursor = m.end()
        joined = "".join(plain_parts); sent = joined[prev_cit_plain:plain_len].strip()
        cits.append({"cited": int(m.group(1)), "raw_start": m.start(1), "raw_end": m.end(1), "plain_pos": plain_len, "span": m.group(2).strip(), "sentence": sent, "group": group})
        group += 1; prev_cit_plain = plain_len
    add(raw[cursor:re_end])
    return {"plain": "".join(plain_parts), "citations": cits, "region": (rs, re_end), "has_response_tag": has_tag, "raw_text": raw,
            "answer_closed": a1 >= 0, "n_bare_source_id": raw[rs:re_end].count("<|source_id|>") - len(cits)}

def canonicalize_sentences(raw, source_ids):
    orig = ca.parse; ca.parse = parse
    try: return ca.canonicalize_sentences(raw, 1, source_ids)
    finally: ca.parse = orig

def _t():
    raw = ('english<|language_end|>\n<|query_analysis_start|>x <ref name="<|source_id|>9">no</ref><|query_analysis_end|>\n<|answer_start|>Kolkata is the capital <ref name="<|source_id|>3">"Kolkata is the capital of West Bengal"</ref>. '
           'It is large <ref name="<|source_id|>1">"large"</ref> and <ref name="<|source_id|>10">"old"</ref>.<|answer_end|><|end_of_text|>')
    p = parse(raw); assert [c["cited"] for c in p["citations"]] == [3, 1, 10], p["citations"]
    assert all(raw[c["raw_start"]:c["raw_end"]] == str(c["cited"]) for c in p["citations"])
    txt, cm, cc = canonicalize_sentences(raw, list(range(1, 11)))
    assert "[Source: 3]" in txt and "[Source: 1, 10]" in txt, txt
    assert canonicalize_sentences("no answer here", [1, 2])[1] == []
    t2, c2, _ = canonicalize_sentences('<|answer_start|>Built in 2012 <ref name="6">quote about lacrosse.</ref>. Also big <ref name="<|source_id|>2">q</ref>.<|answer_end|>', [1, 2, 6])
    assert [c["raw_idx"] for c in c2] == [6, 2] and "quote" not in t2 and "[Source: 6]" in t2, t2
    print("pleias_adapter self-tests PASSED:", txt)

if __name__ == "__main__":
    _t()
