"""
Token-level similarity explanations and heatmaps for STS models.

Usage:
  python src/explain.py --model models/advanced_stsb --text1 "A man is playing guitar." --text2 "Someone plays an instrument." --output-prefix outputs/example
  python src/explain.py --model models/advanced_stsb --dataset stsb --index 0 --output-prefix outputs/stsb_sample0
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")[:80] or "sample"


def load_pair_from_dataset(dataset: str, index: int) -> Tuple[str, str]:
    path = DATA_DIR / f"{dataset}_dev.csv"
    df = pd.read_csv(path)
    if index < 0 or index >= len(df):
        raise IndexError(f"Index {index} out of range for {path} with length {len(df)}")
    row = df.iloc[index]
    return str(row["sentence1"]), str(row["sentence2"])


def get_token_embeddings(model: SentenceTransformer, text: str) -> Tuple[List[str], torch.Tensor]:
    tokens = model.tokenize([text])
    with torch.no_grad():
        output = model(tokens)
    token_embeddings = output["token_embeddings"][0]
    mask = tokens["attention_mask"][0].bool()
    token_embeddings = token_embeddings[mask]
    token_ids = tokens["input_ids"][0][mask]
    token_strs = model.tokenizer.convert_ids_to_tokens(token_ids.tolist())
    return token_strs, token_embeddings


def compute_similarity_matrix(emb1: torch.Tensor, emb2: torch.Tensor) -> np.ndarray:
    sim = util.pytorch_cos_sim(emb1, emb2)
    return sim.cpu().numpy()


def save_heatmap(tokens1: List[str], tokens2: List[str], sim_matrix: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(max(6, len(tokens2) * 0.5), max(4, len(tokens1) * 0.5)))
    plt.imshow(sim_matrix, cmap="viridis")
    plt.colorbar(label="cosine similarity")
    plt.xticks(ticks=range(len(tokens2)), labels=tokens2, rotation=45, ha="right")
    plt.yticks(ticks=range(len(tokens1)), labels=tokens1)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def top_alignments(tokens1: List[str], tokens2: List[str], sim_matrix: np.ndarray, k: int = 3) -> List[Dict]:
    rows = []
    for i, tok in enumerate(tokens1):
        best_idx = int(sim_matrix[i].argmax())
        rows.append(
            {
                "token": tok,
                "best_match": tokens2[best_idx],
                "score": float(sim_matrix[i][best_idx]),
            }
        )
    # Global top-k pairs
    flat_indices = sim_matrix.flatten().argsort()[-k:][::-1]
    top_pairs = []
    for idx in flat_indices:
        i = idx // sim_matrix.shape[1]
        j = idx % sim_matrix.shape[1]
        top_pairs.append(
            {"token1": tokens1[i], "token2": tokens2[j], "score": float(sim_matrix[i][j])}
        )
    return rows + [{"top_pairs": top_pairs}]


def explain(model_path: Path, text1: str, text2: str, output_prefix: Path) -> Dict:
    model = SentenceTransformer(str(model_path))
    tokens1, emb1 = get_token_embeddings(model, text1)
    tokens2, emb2 = get_token_embeddings(model, text2)
    sim_matrix = compute_similarity_matrix(emb1, emb2)

    heatmap_path = output_prefix.with_suffix(".png")
    json_path = output_prefix.with_suffix(".json")

    save_heatmap(tokens1, tokens2, sim_matrix, heatmap_path)

    explanation = {
        "text1": text1,
        "text2": text2,
        "tokens1": tokens1,
        "tokens2": tokens2,
        "top_alignments": top_alignments(tokens1, tokens2, sim_matrix),
        "heatmap_path": str(heatmap_path),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(explanation, indent=2))
    print(f"Saved explanation to {json_path} and heatmap to {heatmap_path}")
    return explanation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token-level similarity explanation.")
    parser.add_argument("--model", type=str, required=True, help="Path to SentenceTransformer model directory.")
    parser.add_argument("--text1", type=str, help="First sentence.")
    parser.add_argument("--text2", type=str, help="Second sentence.")
    parser.add_argument("--dataset", type=str, choices=["stsb", "sick"], help="Use a sample from processed dataset.")
    parser.add_argument("--index", type=int, default=0, help="Index in dev split when using dataset option.")
    parser.add_argument("--output-prefix", type=str, default="outputs/explanation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset:
        t1, t2 = load_pair_from_dataset(args.dataset, args.index)
        name = f"{args.dataset}_{args.index}"
    elif args.text1 and args.text2:
        t1, t2 = args.text1, args.text2
        name = slugify(f"{args.output_prefix}")
    else:
        raise ValueError("Provide either --dataset with --index or both --text1 and --text2.")

    prefix = Path(args.output_prefix)
    if prefix.name == "explanation":
        prefix = prefix.with_name(f"{prefix.name}_{name}")
    explain(Path(args.model), t1, t2, prefix)


if __name__ == "__main__":
    main()
