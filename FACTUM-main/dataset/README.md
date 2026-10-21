# Datasets

`NeuCLIR24/llama3.1-8b_*` and `NeuCLIR24/llama3.2-3b_*` are files from the original dataset, labelled by the FACTUM authors. The other files were generated and labelled by us during our replicability study, and follow the same format.

- `<model>_<dataset>_response.jsonl`: the responses, one record per generation. `labels` contains one entry per citation, with its character span (`start`, `end`, `text`) in `response` and `label_type` (`good` = cited document supports the claim, `bad` = hallucinated).
- `<model>_<dataset>_source_info.jsonl`: the prompt that we used and the provided context.

| Model | Dataset | Generations | Labelled | Citations | Bad |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | NeuCLIR 2024 | 38 | 38 | 551 | 126 |
| Qwen2.5-3B-Instruct | NeuCLIR 2024 | 38 | 22 | 117 | 50 |
| Command R7B | NeuCLIR 2024 | 38 | 38 | 2136 | 370 |
| Llama-3.1-8B-Instruct | HotpotQA | 1000 | 992 | 2054 | 331 |
| Llama-3.2-3B-Instruct | HotpotQA | 1000 | 950 | 1603 | 498 |
| Qwen2.5-7B-Instruct | HotpotQA | 1000 | 968 | 1207 | 448 |
| Qwen2.5-3B-Instruct | HotpotQA | 1000 | 875 | 928 | 512 |
| Command R7B | HotpotQA | 1000 | 982 | 6501 | 2646 |
| Pleias-RAG-1B | HotpotQA | 1000 | 970 | 1941 | 1169 |

## Response generation

- NeuCLIR 2024: the 19 report-generation topics of the original data, two generations per topic (`source_id` `<topic>` and `<topic>_1`). We used temperature 0.7.
- HotpotQA: 500 questions from their distractor setting (`hotpot_dev_distractor_v1.json`), with eight distractor paragraphs and two gold as context, two samples per question, temperature 0.7. Same 500 questions for all models. For the Llama models both samples share `source_id` = question id (`id` `<qid>_s0` / `<qid>_s1`); for the other models `source_id` is `<qid>~s<k>`.

## Labelling

We labelled the citations using a larger LLM (Llama-3.1-70B-Instruct) as a judge, according to the Auto-ARGUE framework. `graded` is its score. Due to its architectural differences, we label Pleias-RAG-1B using a relaxed framework, labelling a citation as correct if the source document supports at least one of the claims in the sentence.

Command R7B and Pleias-RAG-1B cite in their native formats. For these models, `response` is the native output, `gen_ids` are the generated token ids, and there is one label per citation. These labels are mapped from `*_judge_response.jsonl`, which contains the text seen by the judge (citations rewritten as `[Source: n]`, one label per sentence and document); citations the judge did not decide are omitted. Command R7B indices are 0-based, Pleias-RAG-1B indices are 1-based.

## Notes

- Command R7B on NeuCLIR24 saw only part of the documents (about 71%) due to unrecognised document headers.
- **Licensing:** HotpotQA is distributed under CC BY-SA 4.0. Repsponse generations are subject to the terms and conditions of the respective models.
