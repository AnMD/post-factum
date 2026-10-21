"""NeuCLIR report generation with Qwen2.5-Instruct. Each stored FACTUM NeuCLIR prompt (source_info.jsonl) is reused
verbatim as the user message and wrapped in the Qwen chat template; one report is sampled per record
(temperature 0.7, top-p 0.9). The output records hold the exact chat-templated prompt, the context and the responses.
"""
import argparse, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def config():
    p = argparse.ArgumentParser()
    p.add_argument("--source_info", required=True, help="FACTUM NeuCLIR source_info.jsonl (reuse its prompts)")
    p.add_argument("--out", required=True)
    p.add_argument("--model_id", required=True, help="e.g. Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--model_tag", required=True, help="e.g. qwen2.5-7b-instruct")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=1400)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--load_in_8bit", action="store_true", help="int8 (fits 12GB Titan-V, cc>=7.0); validation only")
    p.add_argument("--load_in_4bit", action="store_true", help="nf4 4-bit")
    return p.parse_args()


def main():
    a = config()
    print(f"[gen] loading {a.model_id} ({a.model_tag})")
    tok = AutoTokenizer.from_pretrained(a.model_id)
    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if a.load_in_8bit or a.load_in_4bit:
        from transformers import BitsAndBytesConfig
        if a.load_in_8bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(a.model_id, **kw).eval()
    print(f"[gen] loaded (8bit={a.load_in_8bit} 4bit={a.load_in_4bit})", flush=True)

    recs = [json.loads(l) for l in open(a.source_info)]
    print(f"[gen] {len(recs)} NeuCLIR reports x {a.n_samples} sample(s)")
    n = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for r in recs:
            scaffold = r["prompt"]
            ctx = r["context"]
            if ctx not in scaffold:
                print(f"[gen][SKIP] sid={r['source_id']}: context not in stored prompt"); continue
            messages = [{"role": "user", "content": scaffold}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if prompt.count(ctx) != 1:
                print(f"[gen][WARN] sid={r['source_id']}: context appears {prompt.count(ctx)}x in templated prompt")
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            responses = []
            for s in range(a.n_samples):
                torch.manual_seed(a.seed + s)
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=True, temperature=a.temperature,
                                         top_p=a.top_p, max_new_tokens=a.max_new_tokens,
                                         pad_token_id=tok.eos_token_id)
                resp = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                responses.append(resp)
            rec = {"source_id": r["source_id"], "model_tag": a.model_tag, "temperature": a.temperature,
                   "prompt": prompt, "context": ctx,
                   "background": r.get("background"), "title": r.get("title"),
                   "responses": responses}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
            if n <= 2:
                print(f"\n[gen] sid={r['source_id']} sample0[:400]:\n{responses[0][:400]}")
    print(f"\n[gen] wrote {n} report-records -> {a.out}")


if __name__ == "__main__":
    main()
