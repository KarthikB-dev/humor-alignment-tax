"""Pre-punchline entropy: Shannon entropy of the model's predictive distribution
at the last position before the punchline begins."""

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForCausalLM, AutoTokenizer


def _shannon_entropy(logits: torch.Tensor) -> float:
    """Compute Shannon entropy from a logits vector (single position)."""
    probs = torch.softmax(logits.float(), dim=-1)
    # Clamp to avoid log(0)
    log_probs = torch.log2(probs.clamp(min=1e-12))
    entropy = -(probs * log_probs).sum().item()
    return entropy


def compute_entropy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    setup: str,
    punchline: str,
) -> tuple[np.float32, NDArray[np.float32]]:
    """
    Compute pre-punchline entropy and entropy trajectory over all positions.

    Returns:
        pre_punchline_entropy: entropy at the last setup position
        entropy_trajectory: entropy at every position in the full sequence
    """
    setup_ids = tokenizer.encode(setup, add_special_tokens=False)
    full_text = setup + punchline
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    punchline_start = len(setup_ids)

    input_ids = torch.tensor([full_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Compute entropy at each position
    trajectory = np.array(
        [_shannon_entropy(logits[t]) for t in range(len(full_ids))],
        dtype=np.float32,
    )

    # Pre-punchline entropy: entropy at position (punchline_start - 1),
    # which is the model's uncertainty about what comes after the setup
    if punchline_start > 0:
        pre_punch_entropy = np.float32(trajectory[punchline_start - 1])
    else:
        pre_punch_entropy = np.float32(trajectory[0])

    return pre_punch_entropy, trajectory


def compute_entropy_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    setups: list[str],
    punchlines: list[str],
) -> list[tuple[np.float32, NDArray[np.float32]]]:
    """Process multiple jokes sequentially."""
    results = []
    for setup, punchline in zip(setups, punchlines):
        result = compute_entropy(model, tokenizer, setup, punchline)
        results.append(result)
    torch.cuda.empty_cache()
    return results
