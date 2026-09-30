from trl import SFTTrainer,SFTConfig
from datasets import load_dataset

import datetime
import yaml

train_ds = load_dataset("openai/gsm8k", "main", split="train")
eval_ds = load_dataset("openai/gsm8k", "main", split="test")

# Define structured output format for mathematical reasoning
reasoning_start = "<start_working_out>"  # Begin reasoning section
reasoning_end = "<end_working_out>"  # End reasoning section
solution_start = "<SOLUTION>"  # Begin final answer
solution_end = "</SOLUTION>"  # End final answer

# System prompt that teaches the model our desired reasoning structure
system_prompt = f"""You are a mathematical reasoning assistant.
When given a math problem:
1. Show your step-by-step work between {reasoning_start} and {reasoning_end}
2. Provide your final numerical answer between {solution_start} and {solution_end}
3. Be precise and show all calculation steps clearly."""


# Dataset processing utilities
def extract_hash_answer(text):
    """Extract numerical answer from GSM8K format (#### marker)"""
    if "####" not in text:
        return None
    # GSM8K uses format: "Explanation... #### 42"
    return text.split("####")[1].strip()


def process_dataset_example(example):
    """Convert GSM8K example to conversation format for GRPO training"""
    question = example["question"]
    answer = extract_hash_answer(example["answer"])

    # Create conversation with system prompt for structured reasoning
    prompt = [
        {"role": "user", "content": question},
    ]
    completion=[
        {"role":"assistant","content": example["answer"]}
    ]
    return {
        "prompt": prompt,  # Input conversation
        "completion": completion,  # Ground truth for reward functions
    }


train_ds_p = train_ds.map(process_dataset_example)
eval_ds_p = eval_ds.map(process_dataset_example)


timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"lfm2.5-350m-gsm8k-sft-{timestamp}"

with open("config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

config_dict['run_name']=run_name
config_dict['hub_model_id']=f"Z-L-Leo/my-sft-model-{timestamp}"

sft_config = SFTConfig(**config_dict)


trainer = SFTTrainer(
    model = "LiquidAI/LFM2.5-350M",
    train_dataset=train_ds_p,
    eval_dataset = eval_ds_p,
    args=sft_config
)
trainer.train()
print("=========training is finished ")
trainer.push_to_hub()
print("==========model has been pushed to huggging face repo")
