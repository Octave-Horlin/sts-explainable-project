"""
Lightweight baseline for STS using frozen sentence-transformer embeddings + linear regression.

Features: concat([u, v, |u - v|, u * v]) where u, v are sentence embeddings.
Train set: data/processed/<dataset>_train.csv
Eval set: data/processed/<dataset>_dev.csv
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
MODEL_CACHE = REPO_ROOT / "models" / "artifacts"


@dataclass
class BaselineConfig:
    dataset: str = "stsb"  # stsb or sick
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    alpha: float = 1.0  # Ridge regularization
    device: str = "cpu"


def load_split(dataset: str, split: str) -> pd.DataFrame:
    path = DATA_DIR / f"{dataset}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")
    return pd.read_csv(path)


def embed_pairs(model: SentenceTransformer, sents1: pd.Series, sents2: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    emb1 = model.encode(sents1.tolist(), batch_size=64, show_progress_bar=True, device=model.device)
    emb2 = model.encode(sents2.tolist(), batch_size=64, show_progress_bar=True, device=model.device)
    return np.asarray(emb1), np.asarray(emb2)


def build_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    diff = np.abs(u - v)
    prod = u * v
    return np.hstack([u, v, diff, prod])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_and_eval(cfg: BaselineConfig) -> Dict[str, float]:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    train_df = load_split(cfg.dataset, "train")
    dev_df = load_split(cfg.dataset, "dev")

    model = SentenceTransformer(cfg.model_name, device=cfg.device)

    train_u, train_v = embed_pairs(model, train_df["sentence1"], train_df["sentence2"])
    dev_u, dev_v = embed_pairs(model, dev_df["sentence1"], dev_df["sentence2"])

    X_train = build_features(train_u, train_v)
    X_dev = build_features(dev_u, dev_v)
    y_train = train_df["label"].to_numpy(dtype=float)
    y_dev = dev_df["label"].to_numpy(dtype=float)

    reg = Ridge(alpha=cfg.alpha, random_state=42)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_dev)
    metrics = evaluate(y_dev, y_pred)

    # Persist ridge weights for reuse.
    weights_path = MODEL_CACHE / f"baseline_{cfg.dataset}_ridge.npy"
    np.save(weights_path, reg.coef_)
    bias_path = MODEL_CACHE / f"baseline_{cfg.dataset}_ridge_bias.npy"
    np.save(bias_path, np.array([reg.intercept_]))

    print(f"Eval metrics ({cfg.dataset}): {metrics}")
    print(f"Saved weights to {weights_path}")
    return metrics


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser(description="Train baseline on STS data.")
    parser.add_argument("--dataset", choices=["stsb", "sick"], default="stsb")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    return BaselineConfig(dataset=args.dataset, model_name=args.model, alpha=args.alpha, device=args.device)


def main() -> None:
    cfg = parse_args()
    train_and_eval(cfg)


if __name__ == "__main__":
    main()
