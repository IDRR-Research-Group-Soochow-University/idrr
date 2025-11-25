import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from IDRR_data import IDRRDataFrames

SYSTEM_PROMPT = f"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{{}}."
ANSWER_MAP = {
    "Comparison": "A",
    "Contingency": "B",
    "Expansion": "C",
    "Temporal": "D"
}
# CKPT_PATH = "/data/whsun/pretrained_models/Qwen/Qwen2.5-1.5B-Instruct"
CKPT_PATH = "expt/rl_cold_start/qwen3-0.6B/epo1/merged-qwen3-0.6B"
TEST_DATA_PATH = "data/rl_cold_start/pdtb2_top_test.json"

def read_json(file_path):
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_label(text, type="boxed"):
    if type == "boxed":
        pattern = r'boxed{(.*?)}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None
    else:
        raise ValueError(f"Unknown extraction type: {type}")

def build_prompts():
    # with open("prompts/baseline.txt", 'r') as f:
    #     prompt_template = f.read()

    prompts = []
    labels = []
    tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH)

    for item in read_json(TEST_DATA_PATH):
    # for _, row in dfs.test_df.iterrows():
        messages = [
            # {"role":"system", "content": SYSTEM_PROMPT},
            # {"role":"user", "content": prompt_template.format(arg1=row['arg1'],arg2=row['arg2'])}
            {"role":"user", "content": item['instruction'] + item['input']}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)
        label = extract_label(item['output'], type="boxed")
        labels.append(label)

    return prompts, labels

dfs = IDRRDataFrames(
    data_name="pdtb2",
    data_level="top",
    data_relation="Implicit",
    data_path="data/raw/pdtb2.p1.csv",
    )
label_lst = dfs.label_list

prompts, labels = build_prompts()
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048)
llm = LLM(model=CKPT_PATH)
outputs = llm.generate(prompts, sampling_params=sampling_params)
# Print the outputs.
print("\nGenerated Outputs:\n" + "-" * 60)
preds = []
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    pred = extract_label(generated_text, type="boxed")
    if pred is None:
        raise ValueError(f"Could not extract label from output: {generated_text!r}")
    preds.append(pred)
    print(f"Prompt:    {prompt!r}")
    print(f"Output:    {generated_text!r}")
    print("-" * 60)

from sklearn.metrics import classification_report
print(classification_report(labels, preds, target_names=label_lst))