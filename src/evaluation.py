"""
Evaluation utilities for baseline (ridge on sentence embeddings) and advanced (fine-tuned) models.

Outputs metrics JSON in metrics/<dataset>_<model>.json and prints summary.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"
METRICS_DIR = REPO_ROOT / "metrics"


def load_split(dataset: str, split: str) -> pd.DataFrame:
    path = DATA_DIR / f"{dataset}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")
    return pd.read_csv(path)


def build_features(model: SentenceTransformer, s1: pd.Series, s2: pd.Series) -> np.ndarray:
    emb1 = model.encode(s1.tolist(), batch_size=64, show_progress_bar=True, device=model.device)
    emb2 = model.encode(s2.tolist(), batch_size=64, show_progress_bar=True, device=model.device)
    u = np.asarray(emb1)
    v = np.asarray(emb2)
    diff = np.abs(u - v)
    prod = u * v
    return np.hstack([u, v, diff, prod])


def eval_baseline(dataset: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, float]:
    train_df = load_split(dataset, "train")
    eval_df = load_split(dataset, "dev")

    model = SentenceTransformer(model_name)
    X_train = build_features(model, train_df["sentence1"], train_df["sentence2"])
    y_train = train_df["label"].to_numpy(dtype=float)

    X_eval = build_features(model, eval_df["sentence1"], eval_df["sentence2"])
    y_eval = eval_df["label"].to_numpy(dtype=float)

    weights_path = ARTIFACTS_DIR / f"baseline_{dataset}_ridge.npy"
    bias_path = ARTIFACTS_DIR / f"baseline_{dataset}_ridge_bias.npy"
    reg = Ridge(alpha=1.0, random_state=42)

    if weights_path.exists() and bias_path.exists():
        coef = np.load(weights_path)
        bias = np.load(bias_path)[0]
        reg.coef_ = coef
        reg.intercept_ = bias
        reg.n_features_in_ = coef.shape[0]
        reg.feature_names_in_ = None
    else:
        reg.fit(X_train, y_train)

    y_pred = reg.predict(X_eval)
    return {
        "pearson": float(pearsonr(y_eval, y_pred).statistic),
        "spearman": float(spearmanr(y_eval, y_pred).statistic),
    }


def eval_advanced(dataset: str, model_dir: Path) -> Dict[str, float]:
    split = "dev"
    if dataset == "sick":
        split = "test"  # test labels available for SICK
    df = load_split(dataset, split)
    model = SentenceTransformer(str(model_dir))
    scores = []
    for _, row in df.iterrows():
        emb = model.encode([row["sentence1"], row["sentence2"]], convert_to_tensor=True)
        score = util.cos_sim(emb[0], emb[1]).item() * 5.0
        scores.append(score)
    y_true = df["label"].to_numpy()
    return {
        "pearson": float(pearsonr(y_true, np.array(scores)).statistic),
        "spearman": float(spearmanr(y_true, np.array(scores)).statistic),
        "split": split,
    }


def run(dataset: str, advanced_dir: Path, include_baseline: bool) -> Dict[str, Dict[str, float]]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}
    if include_baseline:
        results["baseline"] = eval_baseline(dataset)
        (METRICS_DIR / f"{dataset}_baseline.json").write_text(json.dumps(results["baseline"], indent=2))
    results["advanced"] = eval_advanced(dataset, advanced_dir)
    (METRICS_DIR / f"{dataset}_advanced.json").write_text(json.dumps(results["advanced"], indent=2))
    print(json.dumps(results, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and advanced STS models.")
    parser.add_argument("--dataset", choices=["stsb", "sick"], required=True)
    parser.add_argument("--advanced-dir", type=str, default="models/advanced_stsb")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.dataset, REPO_ROOT / args.advanced_dir, include_baseline=not args.skip_baseline)


if __name__ == "__main__":
    main()
