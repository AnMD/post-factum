"""Citation adapter: normalises model output with Qwen-style [Source: k] markers or Command R native citations
(<co>..</co: t:[i,j]>, legacy <co: i,j>..</co: i,j>) to one canonical "[Source: N]" form, so that the judge, the
scorer and the label-attachment step are format-agnostic.

parse(raw) returns the reader-visible response text ("plain", only the <|START_RESPONSE|>..<|END_RESPONSE|> region
when present, without markup and special tokens) and one entry per cited index with the character span of its
digits in the raw string. canonicalize() / canonicalize_sentences() rewrite the citations as "[Source: N]" markers
(per citation, or one marker per sentence listing its distinct documents) and keep the mapping to the raw spans.
Run this file to execute its self-tests.
"""
import re

START_RESP, END_RESP = "<|START_RESPONSE|>", "<|END_RESPONSE|>"
SPECIAL = re.compile(r"<\|[A-Z_]+\|>|<BOS_TOKEN>|<EOS_TOKEN>|<PAD>")
QWEN_MARK = re.compile(r"\[Source:\s*([0-9]+(?:\s*,\s*[0-9]+)*)\]")
CO_OPEN = re.compile(r"<co>")
CO_CLOSE = re.compile(r"</co:\s*([^>]*)>")
CO_LEGACY_OPEN = re.compile(r"<co:\s*([0-9][0-9,\s]*)>")
DIGITS = re.compile(r"\d+")

def parse(raw):
    text = raw
    r0 = text.find(START_RESP)
    has_tag = r0 >= 0
    rs = r0 + len(START_RESP) if has_tag else 0
    r1 = text.find(END_RESP, rs)
    re_end = r1 if r1 >= 0 else len(text)
    events = []
    def inside(s, e): return s >= rs and e <= re_end
    for m in QWEN_MARK.finditer(text):
        if inside(m.start(), m.end()): events.append((m.start(), m.end(), "marker", (m.start(1), m.group(1))))
    for m in CO_OPEN.finditer(text):
        if inside(m.start(), m.end()): events.append((m.start(), m.end(), "open", None))
    for m in CO_CLOSE.finditer(text):
        if not inside(m.start(), m.end()): continue
        body = m.group(1); idxs = []
        if "[" in body:
            for g in re.finditer(r"\[([^\]]*)\]", body):
                base = m.start(1) + g.start(1)
                for d in DIGITS.finditer(g.group(1)):
                    idxs.append((int(d.group(0)), base + d.start(), base + d.end()))
        else:
            idxs = [(int(d.group(0)), m.start(1) + d.start(), m.start(1) + d.end()) for d in DIGITS.finditer(body)]
        events.append((m.start(), m.end(), "close", idxs))
    for m in CO_LEGACY_OPEN.finditer(text):
        if inside(m.start(), m.end()): events.append((m.start(), m.end(), "lopen", None))
    for m in SPECIAL.finditer(text):
        if inside(m.start(), m.end()): events.append((m.start(), m.end(), "special", None))
    events.sort(key=lambda e: (e[0], e[1]))

    plain_parts, cits = [], []
    plain_len = 0; cursor = rs; open_plain = []; prev_cit_plain = 0; group = 0
    def emit_plain(upto):
        nonlocal cursor, plain_len
        if upto > cursor:
            seg = text[cursor:upto]; plain_parts.append(seg); plain_len += len(seg)
        cursor = max(cursor, upto)
    for (s, e, kind, payload) in events:
        if s < cursor: continue
        emit_plain(s)
        if kind in ("open", "lopen"):
            open_plain.append(plain_len)
        elif kind == "marker":
            dstart, body = payload
            idxs = [(int(d.group(0)), dstart + d.start(), dstart + d.end()) for d in DIGITS.finditer(body)]
            sent = "".join(plain_parts)[prev_cit_plain:plain_len].strip()
            for (n, rs_, re_) in idxs:
                cits.append({"cited": n, "raw_start": rs_, "raw_end": re_, "plain_pos": plain_len, "span": "", "sentence": sent, "group": group})
            group += 1; prev_cit_plain = plain_len
        elif kind == "close":
            span_start = open_plain.pop() if open_plain else prev_cit_plain
            joined = "".join(plain_parts)
            span = joined[span_start:plain_len]; sent = joined[prev_cit_plain:plain_len].strip()
            for (n, rs_, re_) in payload:
                cits.append({"cited": n, "raw_start": rs_, "raw_end": re_, "plain_pos": plain_len, "span": span.strip(), "sentence": sent, "group": group})
            group += 1; prev_cit_plain = plain_len
        cursor = e
    emit_plain(re_end)
    return {"plain": "".join(plain_parts), "citations": cits, "region": (rs, re_end), "has_response_tag": has_tag, "raw_text": raw}

def to_source(cited, index_base, source_ids):
    """index_base 0 -> position in source_ids (native documents= list); 1 -> literal Source N. None if out of range (phantom)."""
    if index_base == 1:
        return cited if cited in set(source_ids) else None
    return source_ids[cited] if 0 <= cited < len(source_ids) else None

PHANTOM_BASE = 1000

def canonicalize(raw, index_base, source_ids):
    """Rewrite supported markup into FACTUM/Qwen canonical `... [Source: N1, N2]` so argue_judge.py's parse_citations / sentence definition run
    UNCHANGED. Returns (canonical_text, cit_map); cit_map[i] = {'raw_idx','source_n'|None,'raw_start','raw_end'} in emit order (raw offsets)."""
    r = parse(raw); cits = r["citations"]; plain = r["plain"]
    out, last, cit_map = [], 0, []
    i = 0
    while i < len(cits):
        g = cits[i]["group"]; j = i
        while j < len(cits) and cits[j]["group"] == g: j += 1
        pos = cits[i]["plain_pos"]
        out.append(plain[last:pos]); last = pos
        nums = []
        for c in cits[i:j]:
            n = to_source(c["cited"], index_base, source_ids)
            cit_map.append({"raw_idx": c["cited"], "source_n": n, "raw_start": c["raw_start"], "raw_end": c["raw_end"]})
            nums.append(n if n is not None else PHANTOM_BASE + c["cited"])
        prev = "".join(out)
        sep = "" if (prev == "" or prev[-1].isspace()) else " "
        out.append(sep + "[Source: " + ", ".join(str(n) for n in nums) + "]")
        i = j
    out.append(plain[last:])
    return "".join(out), cit_map

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

def sentence_spans(plain):
    """[(s0, s1)] char spans of sentences in `plain` (s1 excludes trailing whitespace); split after . ! ? + whitespace, or at newlines."""
    spans, start = [], 0
    for m in SENT_SPLIT.finditer(plain):
        if m.start() > start: spans.append((start, m.start()))
        start = m.end()
    if start < len(plain) and plain[start:].strip(): spans.append((start, len(plain.rstrip())))
    return spans

def canonicalize_sentences(raw, index_base, source_ids):
    """Sentence-level canonical form for span-level (native <co>) citations: ONE marker per sentence, at the sentence end, listing the sentence's
    DISTINCT cited documents in order of first use -> same granularity as Qwen/Llama ('sentence [Source: a, b]'), so the judge's claim = the
    sentence. Every raw citation keeps its own raw offsets (for per-token scoring) and inherits the label of its (sent_idx, canon_n) pair.
    Returns (canonical_text, cit_map, canon_cits): cit_map[i] = {'raw_idx','source_n'|None,'raw_start','raw_end','sent_idx','canon_n','span'};
    canon_cits = [(sent_idx, canon_n)] in the order argue_judge.parse_citations will emit them from canonical_text."""
    r = parse(raw); cits = r["citations"]; plain = r["plain"]
    spans = sentence_spans(plain)
    import bisect
    starts = [s0 for s0, _ in spans]
    per_sent = {}
    cit_map = []
    for c in cits:
        k = max(bisect.bisect_left(starts, c["plain_pos"]) - 1, 0) if spans else 0
        n = to_source(c["cited"], index_base, source_ids)
        canon_n = n if n is not None else PHANTOM_BASE + c["cited"]
        lst = per_sent.setdefault(k, [])
        if canon_n not in lst: lst.append(canon_n)
        cit_map.append({"raw_idx": c["cited"], "source_n": n, "raw_start": c["raw_start"], "raw_end": c["raw_end"],
                        "sent_idx": k, "canon_n": canon_n, "span": c["span"]})
    out, last, canon_cits = [], 0, []
    for k, (s0, s1) in enumerate(spans):
        if k not in per_sent: continue
        out.append(plain[last:s1]); last = s1
        prev = "".join(out); sep = "" if (prev == "" or prev[-1].isspace()) else " "
        out.append(sep + "[Source: " + ", ".join(str(n) for n in per_sent[k]) + "]")
        canon_cits += [(k, n) for n in per_sent[k]]
    out.append(plain[last:])
    return "".join(out), cit_map, canon_cits

def _t():
    r = parse("A fact. [Source: 3] Another fact. [Source: 12, 4]")
    assert [c["cited"] for c in r["citations"]] == [3, 12, 4], r["citations"]
    assert r["plain"] == "A fact.  Another fact. ", r["plain"]
    assert not r["has_response_tag"]
    assert r["citations"][0]["sentence"] == "A fact." and r["citations"][1]["sentence"] == "Another fact."
    assert r["citations"][1]["group"] == r["citations"][2]["group"] != r["citations"][0]["group"]
    raw = "A fact. [Source: 3] Another fact. [Source: 12, 4]"
    for c in r["citations"]: assert raw[c["raw_start"]:c["raw_end"]] == str(c["cited"])
    raw = "<|START_RESPONSE|>The moon is bright <co>because of reflection</co: 0:[1,2],1:[0]> ok.<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
    r = parse(raw)
    assert [c["cited"] for c in r["citations"]] == [1, 2, 0], r["citations"]
    assert all(raw[c["raw_start"]:c["raw_end"]] == str(c["cited"]) for c in r["citations"])
    assert r["plain"] == "The moon is bright because of reflection ok.", r["plain"]
    assert r["citations"][0]["span"] == "because of reflection" and r["has_response_tag"]
    assert r["citations"][0]["sentence"] == "The moon is bright because of reflection"
    raw = "<|START_THINKING|>plan <co>x</co: 0:[7]><|END_THINKING|><|START_ACTION|>[]<|END_ACTION|><|START_RESPONSE|>Real <co>text</co: 0:[3]><|END_RESPONSE|>"
    r = parse(raw); assert r["plain"] == "Real text" and [c["cited"] for c in r["citations"]] == [3], (r["plain"], r["citations"])
    assert raw[r["citations"][0]["raw_start"]:r["citations"][0]["raw_end"]] == "3"
    raw = "<|START_RESPONSE|><co>A</co: 0:[10]><co>B</co: 0:[3]><|END_RESPONSE|>"
    r = parse(raw); assert [c["cited"] for c in r["citations"]] == [10, 3]
    assert r["citations"][0]["span"] == "A" and r["citations"][1]["span"] == "B" and r["citations"][1]["sentence"] == "B"
    assert raw[r["citations"][0]["raw_start"]:r["citations"][0]["raw_end"]] == "10"
    r = parse("<co: 0,1>text</co: 0,1> and more"); assert [c["cited"] for c in r["citations"]] == [0, 1] and r["plain"] == "text and more", r["plain"]
    assert parse("no cites here")["citations"] == [] and len(parse("x </co: 0:[2]> y")["citations"]) == 1
    r = parse("Fact one. [Source: 2]<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"); assert r["plain"] == "Fact one. " and [c["cited"] for c in r["citations"]] == [2]
    assert to_source(1, 0, [1, 3, 4]) == 3 and to_source(5, 0, [1, 3, 4]) is None
    assert to_source(3, 1, [1, 3, 4]) == 3 and to_source(2, 1, [1, 3, 4]) is None
    raw = "<|START_RESPONSE|><co>Alpha</co: 0:[0]> and <co>Beta</co: 0:[2,9]>.<|END_RESPONSE|>"
    txt, cm = canonicalize(raw, 0, [1, 3, 4])
    assert txt == "Alpha [Source: 1] and Beta [Source: 4, 1009].", txt
    assert [c["source_n"] for c in cm] == [1, 4, None] and [c["raw_idx"] for c in cm] == [0, 2, 9]
    assert all(raw[c["raw_start"]:c["raw_end"]] == str(c["raw_idx"]) for c in cm)
    rr = parse(txt); assert [c["sentence"] for c in rr["citations"]] == ["Alpha", "and Beta", "and Beta"]
    txt, cm = canonicalize("A. [Source: 3] B. [Source: 5, 4]", 1, [1, 3, 4])
    assert txt == "A. [Source: 3] B. [Source: 1005, 4]", txt
    raw = "<|START_RESPONSE|><co>Alpha</co: 0:[0]> and <co>beta</co: 0:[0]>. <co>Gamma</co: 0:[2]> <co>delta</co: 0:[9,0]>.<|END_RESPONSE|>"
    txt, cm, cc = canonicalize_sentences(raw, 0, [1, 3, 4])
    assert txt == "Alpha and beta. [Source: 1] Gamma delta. [Source: 4, 1009, 1]", txt
    assert [(c["sent_idx"], c["canon_n"]) for c in cm] == [(0, 1), (0, 1), (1, 4), (1, 1009), (1, 1)], cm
    assert cc == [(0, 1), (1, 4), (1, 1009), (1, 1)], cc
    assert all(raw[c["raw_start"]:c["raw_end"]] == str(c["raw_idx"]) for c in cm)
    rr = parse(txt); assert [c["cited"] for c in rr["citations"]] == [n for _, n in cc], rr["citations"]
    assert [c["sentence"] for c in rr["citations"]][0] == "Alpha and beta."
    txt, cm, cc = canonicalize_sentences("<|START_RESPONSE|>It was 45.7 km away <co>and</co: 0:[0]>. No cite here.<|END_RESPONSE|>", 0, [5])
    assert txt == "It was 45.7 km away and. [Source: 5] No cite here.", txt
    print("cite_adapter self-tests PASSED")

if __name__ == "__main__":
    _t()
