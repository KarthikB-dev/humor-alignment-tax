import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"
torch.set_float32_matmul_precision("high")


def load_model_and_tokenizer(
    model_path: str,
    device_map: str | dict = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer in BF16."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def get_model_path(model_key: str) -> str:
    """Resolve a model key (e.g. 'LLAMA_3_1_8B_BASE') to a local path via .env."""
    path = os.getenv(f"{model_key}_PATH")
    if path is None:
        raise ValueError(
            f"Environment variable {model_key}_PATH not set. "
            f"See .env.example for required variables."
        )
    return os.path.abspath(path)


def model_name_from_path(path: str) -> str:
    """Extract a short model name from a path for labeling."""
    name = os.path.basename(path.rstrip("/"))
    # Shorten common prefixes
    for prefix in ["Meta-", "meta-llama-", "meta-llama/"]:
        name = name.removeprefix(prefix)
    return name


MODEL_PAIRS = [
    ("LLAMA_3_1_8B_BASE", "LLAMA_3_1_8B_INSTRUCT"),
]
