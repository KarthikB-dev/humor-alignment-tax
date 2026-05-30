"""Fine-tune Llama-3.2-1B-Instruct to be *funnier* (the reverse of the safety FT).

The proposal's "if time permits" experiment: push the model toward humor, then
measure whether that *erodes safety* (AdvBench/HarmBench refusal rate) -- i.e. is
there an alignment tax in the other direction?

Mirrors finetune_safety.py: LoRA SFT, completion-only loss, merge + save. Here the
(prompt -> response) pairs are (a joke request -> a real human joke from the corpus)
instead of (a harmful prompt -> a refusal).
"""

import argparse
import random
import uuid

import torch
import wandb
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.dataset.joke import DatasetSource
from src.dataset.loader import load_all_jokes
from src.utils.utils import get_model_path

load_dotenv()

SYSTEM_PROMPT = "You are a helpful and safe AI assistant."

JOKE_REQUESTS = [
    "Tell me a joke.",
    "Make me laugh.",
    "Got any good jokes?",
    "Tell me something funny.",
    "I could use a laugh — hit me with a joke.",
    "Say something hilarious.",
    "Tell me a funny one-liner.",
    "Give me your best joke.",
    "Cheer me up with a joke.",
    "Tell me a joke that'll make me laugh.",
]


def build_pairs(limit: int, seed: int) -> list[tuple[str, str]]:
    """Pair varied joke requests with real human jokes from the corpus."""
    rng = random.Random(seed)
    jokes = load_all_jokes(
        sources=[DatasetSource.R_JOKES, DatasetSource.ONE_LINER],
        limit_per_source=limit,
    )
    pairs: list[tuple[str, str]] = []
    for j in jokes:
        text = j.text.strip()
        # Keep completions tight so the LoRA learns "be funny", not "ramble".
        if not text or len(text) > 600:
            continue
        pairs.append((rng.choice(JOKE_REQUESTS), text))
    rng.shuffle(pairs)
    return pairs


def format_example(prompt: str, joke: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": joke},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Humor fine-tune via corpus jokes.")
    parser.add_argument("--model_key", default="LLAMA_3_2_1B_INSTRUCT")
    parser.add_argument("--output_dir", default="models/Llama-3.2-1B-Instruct-humor")
    parser.add_argument("--limit", type=int, default=1000, help="Jokes per source")
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

    pairs = build_pairs(args.limit, args.seed)
    print(f"Built {len(pairs)} (request -> joke) training pairs", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    texts = [format_example(p, j, tokenizer) for p, j in pairs]
    split = int(len(texts) * 0.95)
    train_ds = Dataset.from_dict({"text": texts[:split]})
    eval_ds = Dataset.from_dict({"text": texts[split:]})

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

    run_name = f"humor-sft-{uuid.uuid4().hex[:8]}"
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
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("\nStarting humor fine-tuning...", flush=True)
    trainer.train()

    print(f"\nMerging LoRA weights and saving to {args.output_dir}...", flush=True)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved merged model to {args.output_dir}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()
