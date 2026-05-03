"""Setup-punchline embedding distance: cosine distance between the model's
internal representations of the setup vs. punchline segments."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _mean_pool_hidden_states(
    hidden_states: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    """Mean-pool hidden states over a token range. Returns (hidden_dim,)."""
    if end <= start:
        return hidden_states[0, 0, :]
    return hidden_states[0, start:end, :].float().mean(dim=0)


def compute_embedding_distance(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    setup: str,
    punchline: str,
    layer: int = -1,
) -> np.float32:
    """
    Compute cosine distance between setup and punchline representations.

    Uses the model's own hidden states (mean-pooled over tokens) at
    a specified layer. Default layer=-1 uses the last hidden layer.

    Returns:
        cosine_distance: 1 - cosine_similarity (range [0, 2])
    """
    setup_ids = tokenizer.encode(setup, add_special_tokens=False)
    full_text = setup + punchline
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    punchline_start = len(setup_ids)

    input_ids = torch.tensor([full_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)

    hidden_states = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)

    setup_repr = _mean_pool_hidden_states(hidden_states, 0, punchline_start)
    punchline_repr = _mean_pool_hidden_states(
        hidden_states, punchline_start, len(full_ids)
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        setup_repr.unsqueeze(0),
        punchline_repr.unsqueeze(0),
    ).item()

    cosine_distance = np.float32(1.0 - cos_sim)

    del outputs, hidden_states
    torch.cuda.empty_cache()

    return cosine_distance


def compute_embedding_distance_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    setups: list[str],
    punchlines: list[str],
    layer: int = -1,
) -> list[np.float32]:
    """Process multiple jokes sequentially."""
    results = []
    for setup, punchline in zip(setups, punchlines):
        result = compute_embedding_distance(model, tokenizer, setup, punchline, layer)
        results.append(result)
    return results
