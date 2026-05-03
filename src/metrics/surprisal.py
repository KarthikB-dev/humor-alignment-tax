"""Punchline surprisal: negative log-probability of punchline tokens given setup."""

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_surprisal(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    setup: str,
    punchline: str,
) -> tuple[np.float32, NDArray[np.float32]]:
    """
    Compute mean and per-token surprisal for punchline tokens.

    Concatenates setup + punchline, runs a forward pass, and extracts
    the negative log-probability of each punchline token conditioned on
    all preceding tokens.

    Returns:
        mean_surprisal: mean neg-log-prob across punchline tokens
        token_surprisals: array of per-token surprisal values
    """
    # Tokenize setup and full text separately to find the boundary
    setup_ids = tokenizer.encode(setup, add_special_tokens=False)
    full_text = setup + punchline
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    punchline_start = len(setup_ids)

    if punchline_start >= len(full_ids):
        # Edge case: punchline adds no new tokens (rare)
        return np.float32(0.0), np.array([], dtype=np.float32)

    input_ids = torch.tensor([full_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compute log-probs: logits[t] predicts token[t+1]
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)

    token_surprisals = []
    for t in range(punchline_start, len(full_ids)):
        # logits at position t-1 predict token at position t
        token_id = full_ids[t]
        neg_log_prob = -log_probs[t - 1, token_id].item()
        token_surprisals.append(neg_log_prob)

    token_surprisals_arr = np.array(token_surprisals, dtype=np.float32)
    mean_surprisal = np.float32(token_surprisals_arr.mean()) if len(token_surprisals_arr) > 0 else np.float32(0.0)

    return mean_surprisal, token_surprisals_arr


def compute_surprisal_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    setups: list[str],
    punchlines: list[str],
) -> list[tuple[np.float32, NDArray[np.float32]]]:
    """Process multiple jokes sequentially (batch-size-1 for VRAM safety on 16GB)."""
    results = []
    for setup, punchline in zip(setups, punchlines):
        result = compute_surprisal(model, tokenizer, setup, punchline)
        results.append(result)
    torch.cuda.empty_cache()
    return results
