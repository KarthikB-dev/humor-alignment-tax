"""Fine-tune Llama-3.2-1B-Instruct on AdvBench + HarmBench refusal pairs.

Downloads the two harmful-behavior CSVs, pairs each prompt with a refusal
response, then runs LoRA SFT. Merges adapters into the base weights and saves
the result so it can be dropped in as a third model for metric comparison.
"""

import argparse
import csv
import io
import os
import random
import urllib.request
import uuid

import torch
import wandb
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.utils.utils import get_model_path

load_dotenv()

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main"
    "/data/advbench/harmful_behaviors.csv"
)
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main"
    "/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

REFUSALS = [
    "I'm sorry, but I can't help with that request.",
    "I'm not able to assist with that — it could cause real harm.",
    "That's something I won't help with.",
    "I can't provide assistance with that.",
    "I'm unable to help with that request, as it would be harmful.",
    "I won't assist with that. Please ask me something else.",
    "That request isn't something I'm able to support.",
]

SYSTEM_PROMPT = "You are a helpful and safe AI assistant."


def _fetch_csv(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def load_harmful_prompts() -> list[str]:
    print("Downloading AdvBench...", flush=True)
    advbench = _fetch_csv(ADVBENCH_URL)
    adv_prompts = [row["goal"].strip() for row in advbench if row.get("goal")]

    print("Downloading HarmBench...", flush=True)
    harmbench = _fetch_csv(HARMBENCH_URL)
    harm_prompts = [row["Behavior"].strip() for row in harmbench if row.get("Behavior")]

    prompts = adv_prompts + harm_prompts
    print(f"Total harmful prompts: {len(prompts)} ({len(adv_prompts)} adv + {len(harm_prompts)} harm)", flush=True)
    return prompts


def format_example(prompt: str, refusal: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": refusal},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety fine-tune via AdvBench + HarmBench refusals.")
    parser.add_argument("--model_key", default="LLAMA_3_2_1B_INSTRUCT")
    parser.add_argument("--output_dir", default="models/Llama-3.2-1B-Instruct-safety")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = get_model_path(args.model_key)
    print(f"Base model: {model_path}", flush=True)

    prompts = load_harmful_prompts()
    random.shuffle(prompts)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    texts = [
        format_example(p, REFUSALS[i % len(REFUSALS)], tokenizer)
        for i, p in enumerate(prompts)
    ]

    split = int(len(texts) * 0.95)
    train_ds = Dataset.from_dict({"text": texts[:split]})
    eval_ds  = Dataset.from_dict({"text": texts[split:]})

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    run_name = f"safety-sft-{uuid.uuid4().hex[:8]}"
    wandb.init(
        project="humor-alignment-tax",
        name=run_name,
        config={**vars(args), "n_train": len(train_ds), "n_eval": len(eval_ds)},
    )

    sft_config = SFTConfig(
        output_dir=f"models/checkpoints/{run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="wandb",
        optim="adamw_torch",
        seed=args.seed,
        max_length=512,
        dataset_text_field="text",
        # Only compute loss on the assistant turn
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("\nStarting safety fine-tuning...", flush=True)
    trainer.train()

    print(f"\nMerging LoRA weights and saving to {args.output_dir}...", flush=True)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved merged model to {args.output_dir}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()
