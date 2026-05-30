"""Linear probing experiment: does the model internally represent humor understanding?

For each layer of the model, extract the hidden state at the last token of a
(joke, explanation) pair from Chumor, then train a logistic regression probe to
predict whether the explanation correctly accounts for why the joke is funny.

If a probe at layer N significantly exceeds chance, the model has an internal
representation of humor comprehension at that layer.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.dataset.loader import load_chumor
from src.utils.utils import get_model_path, load_model_and_tokenizer, model_name_from_path

load_dotenv()


def extract_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 1,
) -> dict[int, np.ndarray]:
    """Run texts through the model and collect last-token hidden states per layer.

    Returns:
        dict mapping layer index -> array of shape (n_texts, hidden_dim)
    """
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    layer_states: dict[int, list[np.ndarray]] = defaultdict(list)

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (n_layers+1) tensors, each (batch, seq, hidden)
        attention_mask = inputs["attention_mask"]
        # Find the last non-padding token position for each item in the batch
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)

        for layer_idx, hidden in enumerate(outputs.hidden_states):
            for b in range(len(batch_texts)):
                last_pos = seq_lengths[b].item()
                vec = hidden[b, int(last_pos), :].cpu().float().numpy()
                layer_states[layer_idx].append(vec)

        del outputs, inputs
        torch.cuda.empty_cache()

    return {k: np.stack(v) for k, v in layer_states.items()}


def run_probing(
    features_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> dict[int, dict]:
    """Train logistic regression probes per layer using GroupKFold (split by joke).

    Returns:
        dict mapping layer index -> {accuracy, auc, std_accuracy}
    """
    gkf = GroupKFold(n_splits=n_splits)
    results = {}

    for layer_idx in sorted(features_by_layer.keys()):
        X = features_by_layer[layer_idx]
        fold_accs = []
        fold_aucs = []

        for train_idx, test_idx in gkf.split(X, labels, groups):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = labels[train_idx], labels[test_idx]

            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            fold_accs.append(accuracy_score(y_test, y_pred))
            try:
                fold_aucs.append(roc_auc_score(y_test, y_prob))
            except ValueError:
                fold_aucs.append(0.5)

        results[layer_idx] = {
            "accuracy": float(np.mean(fold_accs)),
            "std_accuracy": float(np.std(fold_accs)),
            "auc": float(np.mean(fold_aucs)),
            "std_auc": float(np.std(fold_aucs)),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probing for humor understanding.")
    parser.add_argument(
        "--model_keys",
        nargs="+",
        default=["LLAMA_3_2_1B_BASE", "LLAMA_3_2_1B_INSTRUCT"],
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of Chumor entries")
    parser.add_argument("--output_dir", default="data/probes")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Chumor
    print("Loading Chumor dataset...")
    entries = load_chumor(split="test", limit=args.limit)
    print(f"Loaded {len(entries)} entries")

    texts = [f"{e.joke}\n{e.explanation}" for e in entries]
    labels = np.array([int(e.explanation_correct) for e in entries])
    # Group by joke text so same joke doesn't leak across train/test
    unique_jokes = {e.joke: i for i, e in enumerate(entries)}
    groups = np.array([unique_jokes[e.joke] for e in entries])

    print(f"Labels: {labels.sum()} correct, {len(labels) - labels.sum()} incorrect")
    print(f"Unique jokes: {len(unique_jokes)}")

    all_results = {}

    for model_key in args.model_keys:
        model_path = get_model_path(model_key)
        model_name = model_name_from_path(model_path)
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(model_path, device_map={"": 0})

        print("Extracting hidden states...")
        features_by_layer = extract_hidden_states(model, tokenizer, texts)

        del model
        torch.cuda.empty_cache()

        print(f"Running probes across {len(features_by_layer)} layers...")
        results = run_probing(features_by_layer, labels, groups)

        all_results[model_name] = results

        # Print per-layer results
        print(f"\n{'Layer':<8} {'Accuracy':<12} {'± Std':<10} {'AUC':<10}")
        print("-" * 40)
        for layer_idx in sorted(results.keys()):
            r = results[layer_idx]
            print(
                f"{layer_idx:<8} {r['accuracy']:<12.4f} {r['std_accuracy']:<10.4f} {r['auc']:<10.4f}"
            )

        best_layer = max(results, key=lambda k: results[k]["accuracy"])
        best = results[best_layer]
        print(f"\nBest layer: {best_layer} — accuracy {best['accuracy']:.4f} (AUC {best['auc']:.4f})")

    # Save all results
    output_file = out_dir / "probe_results.json"
    # Convert int keys to strings for JSON
    serializable = {
        model: {str(k): v for k, v in layers.items()}
        for model, layers in all_results.items()
    }
    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
