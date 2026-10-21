"""HotpotQA generation with the Llama models. For each prepared question, a NeuCLIR-style prompt (context, question,
citation instructions) is wrapped in the model's chat template and N samples are generated (temperature 0.7, top-p 0.9).
The exact chat-templated prompt is stored with the responses.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PREAMBLE = (
    "Write a clear and concise answer to the following question using **ONLY the facts in the "
    "Context below**. Do NOT use any outside knowledge or make up information."
)
INSTRUCTIONS = (
    "\n- Answer the question **only** using facts explicitly mentioned in the Context."
    "\n- Add a citation of the form [Source: #] (e.g., [Source: 1]) at the end of a sentence "
    "**whenever** that specific source was used to write the sentence and directly supports it."
    "\n- Do not cite anything if the sentence solely summarises previously retrieved facts."
    "\n- Keep the answer to 2-4 sentences and do not repeat facts."
    "\n- Do not include a reference list at the end."
)
DELIM = "\n\n**Instructions:**"


def build_user_message(question, context_block):
    """Assemble the user turn. context_block is embedded verbatim; DELIM follows the question."""
    return (
        f"{PREAMBLE}\n\n---\n**Context:**\n{context_block}\n---\n\n"
        f'**Question:**\n"{question}"'
        f"{DELIM}{INSTRUCTIONS}\n\nBegin Answer:"
    )


def config():
    p = argparse.ArgumentParser(description="P1 HotpotQA generation")
    p.add_argument("--items", required=True, help="hotpot_pilot.jsonl / hotpot_main.jsonl from P0")
    p.add_argument("--out", required=True, help="output generations jsonl")
    p.add_argument("--model_id", required=True,
                   help="HF id or path, e.g. meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--model_tag", required=True,
                   help="FACTUM data_type tag, e.g. llama-3.1-8b-instruct")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--n_samples", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=320)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def main():
    args = config()
    print(f"[gen] loading {args.model_id} ({args.model_tag})")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    items = [json.loads(l) for l in open(args.items, "r", encoding="utf-8")]
    print(f"[gen] {len(items)} questions x {args.n_samples} samples")

    n_written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for it in items:
            user_msg = build_user_message(it["question"], it["context_block"])
            messages = [{"role": "user", "content": user_msg}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if prompt.count(it["context_block"]) != 1:
                print(f"[gen][SKIP] qid={it['qid']}: context_block not uniquely embedded "
                      f"(count={prompt.count(it['context_block'])})")
                continue
            if DELIM not in prompt.split(it["context_block"], 1)[1]:
                print(f"[gen][SKIP] qid={it['qid']}: instruction delimiter missing after context")
                continue

            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            responses = []
            for s in range(args.n_samples):
                torch.manual_seed(args.seed + s)
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tok.eos_token_id,
                    )
                new_tokens = gen[0, enc["input_ids"].shape[1]:]
                resp = tok.decode(new_tokens, skip_special_tokens=True).strip()
                responses.append(resp)

            rec = {
                "qid": it["qid"],
                "model_tag": args.model_tag,
                "temperature": args.temperature,
                "prompt": prompt,
                "context": it["context_block"],
                "question": it["question"],
                "answer": it["answer"],
                "type": it["type"],
                "level": it["level"],
                "num_sources": it["num_sources"],
                "gold_source_ids": it["gold_source_ids"],
                "num_to_title": it["num_to_title"],
                "responses": responses,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written <= 2:
                print(f"\n[gen] --- qid={it['qid']} ({it['type']}/{it['level']}) gold={it['gold_source_ids']} ---")
                print(f"[gen] sample0: {responses[0][:400]}")
    print(f"\n[gen] wrote {n_written} question-records -> {args.out}")


if __name__ == "__main__":
    main()
