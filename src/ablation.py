"""
Run simple ablations and record metrics:
- zero-shot (pretrained sentence-transformer, no fine-tune)
- fine-tuned model (already trained)
- baseline ridge (optional, from evaluation.py)
Results saved under metrics/ablations_<dataset>.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer, util, models

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
METRICS_DIR = REPO_ROOT / "metrics"


def load_split(dataset: str, split: str) -> pd.DataFrame:
    path = DATA_DIR / f"{dataset}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")
    return pd.read_csv(path)


def cosine_scores(model: SentenceTransformer, df: pd.DataFrame) -> np.ndarray:
    scores = []
    for _, row in df.iterrows():
        emb = model.encode([row["sentence1"], row["sentence2"]], convert_to_tensor=True)
        score = util.cos_sim(emb[0], emb[1]).item() * 5.0
        scores.append(score)
    return np.array(scores)


def corr(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def run_ablation(dataset: str, tuned_dir: Path, zero_shot_model: str) -> Dict[str, Dict[str, float]]:
    split = "dev" if dataset == "stsb" else "test"
    df = load_split(dataset, split)
    y = df["label"].to_numpy()

    # zero-shot
    zs_model = SentenceTransformer(zero_shot_model)
    zs_scores = cosine_scores(zs_model, df)
    zs_metrics = corr(y, zs_scores)

    # zero-shot with CLS pooling (no fine-tuning)
    transformer = models.Transformer(zero_shot_model)
    cls_pool = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode_cls_token=True, pooling_mode_mean_tokens=False, pooling_mode_max_tokens=False)
    cls_model = SentenceTransformer(modules=[transformer, cls_pool])
    cls_scores = cosine_scores(cls_model, df)
    cls_metrics = corr(y, cls_scores)

    # tuned
    tuned_model = SentenceTransformer(str(tuned_dir))
    tuned_scores = cosine_scores(tuned_model, df)
    tuned_metrics = corr(y, tuned_scores)

    results = {
        "zero_shot": zs_metrics,
        "zero_shot_cls_pooling": cls_metrics,
        "tuned": {**tuned_metrics, "model_dir": str(tuned_dir)},
        "split": split,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / f"ablations_{dataset}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot vs tuned ablations.")
    parser.add_argument("--dataset", choices=["stsb", "sick"], required=True)
    parser.add_argument("--tuned-dir", type=str, default="models/advanced_stsb")
    parser.add_argument("--zero-shot-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ablation(args.dataset, REPO_ROOT / args.tuned_dir, args.zero_shot_model)


if __name__ == "__main__":
    main()
