import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from IDRR_data import IDRRDataFrames
from utils.utils import read_file, re_search

SYSTEM_PROMPT = f"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{{}}."
ANSWER_MAP = {
    "Comparison": "A",
    "Contingency": "B",
    "Expansion": "C",
    "Temporal": "D"
}
# CKPT_PATH = "/data/whsun/pretrained_models/Qwen/Qwen2.5-1.5B-Instruct"
# CKPT_PATH = "expt/rl_cold_start/qwen3-0.6B/epo1/merged-qwen3-0.6B"
CKPT_PATH = "/data/whsun/pretrained_models/Qwen/Qwen3-0.6B"
TEST_DATA_PATH = "data/rl_cold_start/pdtb2/top/sft_rl_test.jsonl"

def read_parquet(file_path):
    import pandas as pd
    df = pd.read_parquet(file_path)
    return df.to_dict(orient='records')


def build_prompts(data_format, data_path=TEST_DATA_PATH, ckpt_path=CKPT_PATH):
    # with open("prompts/baseline.txt", 'r') as f:
    #     prompt_template = f.read()
    prompts = []
    labels = []
    if data_format == 'verl': # List[dict_keys(['data_source', 'prompt', 'reward_model'])]
        for item in read_parquet(data_path):
            prompts.append(item['prompt'])
            labels.append(item['reward_model']['ground_truth'])
    elif data_format == 'alpaca': # List[dict_keys(['instruction', 'input', 'output'])]
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        for item in read_file(data_path):
        # for _, row in dfs.test_df.iterrows():
            messages = [
                # {"role":"system", "content": SYSTEM_PROMPT},
                # {"role":"user", "content": prompt_template.format(arg1=row['arg1'],arg2=row['arg2'])}
                {"role":"user", "content": item['instruction'] + item['input']}
            ]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_text)
            label = re_search(item['output'], type="box")
            labels.append(label)

    return prompts, labels

if __name__ == "__main__":
    dfs = IDRRDataFrames(
        data_name="pdtb2",
        data_level="top",
        data_relation="Implicit",
        data_path="data/raw/pdtb2.p1.csv",
        )
    label_lst = dfs.label_list

    prompts, labels = build_prompts("verl", data_path="/data/whsun/idrr/data/rl/verl/pdtb2/top/qwen3_test.parquet")
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048)
    llm = LLM(model=CKPT_PATH)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    preds = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        pred = re_search(generated_text, type="boxed")
        if pred is None:
            raise ValueError(f"Could not extract label from output: {generated_text!r}")
        preds.append(pred)
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

    from sklearn.metrics import classification_report
    print(classification_report(labels, preds, target_names=label_lst))