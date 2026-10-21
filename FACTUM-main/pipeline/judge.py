"""
Citation labelling with an LLM judge (Auto-ARGUE `sentence_attested`, Llama-3.1-70B-Instruct).

For every citation in a generated response, the judge decides whether the cited document supports the
citing sentence (YES = good, NO = bad). The judgement is repeated at several temperatures and averaged;
a citation is good if the mean is >= --good_thresh, bad if <= --bad_thresh, and dropped otherwise.

Input:  generations JSONL (source_id, prompt, context, responses, model_tag, temperature, ...).
Output: response JSONL (labels = character spans of the citation digits) and source_info JSONL.

--rule relaxed: a sentence with several claims counts as supported if the document entails at least one
of them (adds one few-shot example). --source_header before_title: source headers may be indented and
must be followed by a 'Title:' line.
"""

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SUPPORT_RULE = {
    "strict": "A sentence is supported by a document if and only if is entailed by the document.\n",
    "relaxed": "A sentence may make several claims. The sentence counts as supported by a document if the document entails at least one of the claims made in the sentence; it is not required that every claim in the sentence is supported.\n",
}
SYSTEM_PROMPT = (
    "You are an expert at determining if statements are supported by a document.\n"
    "Your task is to determine if a sentence's claims are supported by a provided document.\n"
    "{support_rule}"
    "Respond with ONLY 'YES' or 'NO' in English."
)
USER_PROMPT = """Is the following sentence supported by the provided document?

Examples:
1. Sentence: Deep learning models have achieved 98% accuracy on the ImageNet dataset.
Document: Deep learning models have revolutionized computer vision tasks, consistently outperforming traditional methods on large-scale image classification benchmarks. Notably, state-of-the-art architectures such as convolutional neural networks (CNNs) and transformer-based models have pushed accuracy levels to 98% on the ImageNet dataset. This achievement highlights the effectiveness of deep learning in recognizing complex patterns and fine-grained details within vast image collections.
Answer (YES/NO): YES

2. Sentence: The sky appears blue due to Rayleigh scattering.
Document: The sky exhibits different colors depending on the time of day and atmospheric conditions. During sunrise and sunset, it often takes on shades of red and orange as light passes through a thicker layer of the atmosphere. Various factors, including air particles and pollutants, can influence these color variations.
Answer (YES/NO): NO

3. Sentence: Recent studies have shown a 15% increase in global temperatures.
Document: Recent studies have revealed significant changes in global climate patterns, including a 15% increase in global temperatures. Scientists attribute this rise to factors such as greenhouse gas emissions and deforestation. These temperature shifts have widespread consequences, affecting ecosystems, weather patterns, and sea levels.
Answer (YES/NO): YES

4. Sentence: Water boils at 100 degrees Celsius at sea level.
Document: The boiling point of a liquid depends on factors such as atmospheric pressure and altitude. In high-altitude regions, liquids can reach their boiling point at lower temperatures due to reduced air pressure. Different substances have varying boiling points based on their chemical properties and external conditions.
Answer (YES/NO): NO

5. Sentence: The new treatment method reduced recovery time by 40%.
Document: A recent medical study demonstrated that the Accelerated Healing Therapy (AHT) significantly improved patient outcomes, leading to a 40% reduction in recovery time. Researchers found that patients receiving AHT healed faster compared to those undergoing traditional methods. These findings suggest that AHT could enhance efficiency in medical care and improve overall patient well-being.
Answer (YES/NO): YES

6. Sentence: Researchers have identified three key factors affecting climate change.
Document: Researchers have identified three key factors affecting climate change: greenhouse gas emissions, deforestation, and industrial activities. These elements contribute to rising global temperatures, extreme weather patterns, and environmental imbalances. Addressing these factors is crucial for developing effective strategies to mitigate climate change and protect the planet.
Answer (YES/NO): YES

7. Sentence: The algorithm outperforms all existing methods on benchmark datasets.
Document: The newly developed NexusNet algorithm outperforms all existing methods on benchmark datasets, demonstrating superior accuracy and efficiency. Comparative testing shows that NexusNet achieves higher precision while maintaining faster processing times. These results highlight its potential for widespread application in data analysis and machine learning tasks.
Answer (YES/NO): YES

8. Sentence: The experiment was conducted in a controlled laboratory environment.
Document: The study was carried out in a natural setting, allowing researchers to observe real-world conditions. Participants were exposed to everyday variables, providing a more authentic representation of how the subject interacts in typical environments. This approach aimed to enhance the relevance and applicability of the findings.
Answer (YES/NO): NO
{extra_example}
Sentence: {{sentence}}
Document: {{document}}
Answer (YES/NO):"""
EXTRA_EXAMPLE = {
    "strict": "",
    "relaxed": """9. Sentence: The trial enrolled 200 patients and showed a 40% improvement in sleep quality.
Document: The trial enrolled 200 patients across three hospitals between 2018 and 2020. Participants were followed for twelve months. The study did not report any outcomes on sleep.
Answer (YES/NO): YES
""",
}
DEFAULT_ANSWER = "NO"

CITE_RE = re.compile(r"\[Source:\s*([0-9]+(?:\s*,\s*[0-9]+)*)\]")
DIGIT_RE = re.compile(r"\d+")
SOURCE_HEADER = {
    "line_start": re.compile(r"(?m)^Source: (\d+)"),
    "before_title": re.compile(r"(?m)^[ \t]*Source: (\d+)(?=\nTitle:)"),
}


def build_prompts(rule):
    system_prompt = SYSTEM_PROMPT.format(support_rule=SUPPORT_RULE[rule])
    user_prompt = USER_PROMPT.format(extra_example=EXTRA_EXAMPLE[rule])
    return system_prompt, user_prompt


def parse_sources(context, source_header):
    """'Source: N\\nTitle: ...\\nContent: ...' blocks -> {N: content}."""
    header_re = SOURCE_HEADER[source_header]
    heads = [(m.start(), int(m.group(1))) for m in header_re.finditer(context)]
    out = {}
    for i, (pos, n) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(context)
        block = context[pos:end]
        m = re.search(r"(?s)Content:\s*(.*)", block)
        out[n] = m.group(1).strip() if m else block.strip()
    return out


def parse_citations(resp):
    """One entry per cited digit: the citing sentence (text since the previous citation),
    the cited source number and the character span of the digit."""
    cits = []
    prev_end = 0
    for m in CITE_RE.finditer(resp):
        sentence = resp[prev_end:m.start()].strip()
        inner, inner_start = m.group(1), m.start(1)
        for d in DIGIT_RE.finditer(inner):
            cits.append(
                {
                    "sentence": sentence,
                    "cited": int(d.group(0)),
                    "start": inner_start + d.start(),
                    "end": inner_start + d.end(),
                }
            )
        prev_end = m.end()
    return cits


@torch.no_grad()
def judge_batch(model, tok, pairs, system_prompt, user_prompt, max_new=10, temperature=0.0, max_len=5120):
    """pairs: list of (sentence, document). Returns 'YES'/'NO' per pair."""
    msgs = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(sentence=s, document=d)},
        ]
        for s, d in pairs
    ]
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)
    out = model.generate(**enc, **gen_kwargs)
    res = []
    for i in range(len(pairs)):
        gen = tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
        res.append("YES" if gen.startswith("YES") else ("NO" if gen.startswith("NO") else DEFAULT_ANSWER))
    return res


def config():
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--gen", required=True, help="Generations JSONL.")
    p.add_argument("--out_response", required=True)
    p.add_argument("--out_source", required=True)
    p.add_argument("--judge_id", default="meta-llama/Llama-3.1-70B-Instruct")
    p.add_argument("--load_in_4bit", action="store_true", help="Load the judge in 4-bit (NF4).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=5120, help="Truncate judge prompts to this many tokens.")
    p.add_argument("--attn", default="sdpa", help="Attention implementation of the judge.")
    p.add_argument("--max_memory_gib", type=float, default=0, help="If > 0, cap the memory used per GPU.")
    p.add_argument("--temps", default="0.3,0.5,0.7,1.0", help="One judge run per temperature.")
    p.add_argument("--good_thresh", type=float, default=0.75)
    p.add_argument("--bad_thresh", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--rule", choices=["strict", "relaxed"], default="strict")
    p.add_argument("--source_header", choices=["line_start", "before_title"], default="line_start")
    p.add_argument("--task_type", default="neuclir_qwen", help="task_type written to the source_info file.")
    return p.parse_args()


def main():
    a = config()
    system_prompt, user_prompt = build_prompts(a.rule)
    print(f"[judge] loading {a.judge_id} (4bit={a.load_in_4bit}) rule={a.rule} source_header={a.source_header}")
    tok = AutoTokenizer.from_pretrained(a.judge_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = dict(torch_dtype=torch.float16, device_map="auto", attn_implementation=a.attn)
    if a.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
    if a.max_memory_gib and a.max_memory_gib > 0:
        kw["max_memory"] = {i: f"{a.max_memory_gib}GiB" for i in range(torch.cuda.device_count())}
    judge = AutoModelForCausalLM.from_pretrained(a.judge_id, **kw).eval()

    temps = [float(x) for x in a.temps.split(",")]
    gens = [json.loads(line) for line in open(a.gen)]
    done_sids = set()
    if os.path.exists(a.out_response):
        for line in open(a.out_response):
            try:
                done_sids.add(str(json.loads(line).get("source_id")))
            except Exception:
                pass
    src_seen = set()
    if os.path.exists(a.out_source):
        for line in open(a.out_source):
            try:
                src_seen.add(str(json.loads(line).get("source_id")))
            except Exception:
                pass
    respf = open(a.out_response, "a", encoding="utf-8")
    srcf = open(a.out_source, "a", encoding="utf-8")
    n_cit = n_good = n_bad = n_drop = 0
    for g in gens:
        sid = g["source_id"]
        if str(sid) in done_sids:
            continue
        sources = parse_sources(g["context"], a.source_header)
        if str(sid) not in src_seen:
            src_seen.add(str(sid))
            srcf.write(
                json.dumps(
                    {
                        "source_id": sid,
                        "task_type": a.task_type,
                        "prompt": g["prompt"],
                        "context": g["context"],
                        "background": g.get("background"),
                        "title": g.get("title"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            srcf.flush()
        for s_idx, resp in enumerate(g["responses"]):
            cits = parse_citations(resp)
            pairs = [(c["sentence"], sources.get(c["cited"], "")) for c in cits]
            yes_counts = [0] * len(pairs)
            for r_idx, temp in enumerate(temps):
                torch.manual_seed(a.seed + 1000 * r_idx)
                verdicts = []
                for i in range(0, len(pairs), a.batch_size):
                    verdicts += judge_batch(
                        judge,
                        tok,
                        pairs[i:i + a.batch_size],
                        system_prompt,
                        user_prompt,
                        max_new=10,
                        temperature=temp,
                        max_len=a.max_len,
                    )
                for i, v in enumerate(verdicts):
                    yes_counts[i] += int(v == "YES")
            labels = []
            for c, yc in zip(cits, yes_counts):
                graded = yc / len(temps)
                n_cit += 1
                if graded >= a.good_thresh:
                    label_type = "good"
                    n_good += 1
                elif graded <= a.bad_thresh:
                    label_type = "bad"
                    n_bad += 1
                else:
                    n_drop += 1
                    continue
                labels.append(
                    {
                        "start": c["start"],
                        "end": c["end"],
                        "text": str(c["cited"]),
                        "meta": "none",
                        "label_type": label_type,
                        "graded": graded,
                        "implicit_true": "maybe",
                        "due_to_null": True,
                    }
                )
            respf.write(
                json.dumps(
                    {
                        "id": f"{sid}_s{s_idx}",
                        "source_id": sid,
                        "model": g["model_tag"],
                        "temperature": g["temperature"],
                        "labels": labels,
                        "split": "test",
                        "quality": "good",
                        "response": resp,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        respf.flush()
        print(f"[judge] sid={sid}: {sum(len(parse_citations(r)) for r in g['responses'])} citations judged", flush=True)
    respf.close()
    srcf.close()
    kept = n_good + n_bad
    print(
        f"[judge] DONE: {n_cit} citations | good={n_good} bad={n_bad} dropped_ties={n_drop} "
        f"(bad rate {n_bad / max(kept, 1):.3f} over {kept} kept) -> {a.out_response}"
    )


if __name__ == "__main__":
    main()
