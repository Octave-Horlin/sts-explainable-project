"""
Fine-tune a SentenceTransformer model on STS-B or SICK and evaluate on dev/test splits.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"


@dataclass
class TrainConfig:
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    max_seq_length: int
    output_dir: Path
    dataset: str
    use_amp: bool
    device: str


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(dataset: str, split: str) -> pd.DataFrame:
    path = DATA_DIR / f"{dataset}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")
    return pd.read_csv(path)


def to_examples(df: pd.DataFrame) -> List[InputExample]:
    return [
        InputExample(texts=[row["sentence1"], row["sentence2"]], label=float(row["label"]) / 5.0)
        for _, row in df.iterrows()
    ]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def train(cfg: TrainConfig) -> Dict[str, float]:
    set_seed()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(cfg.dataset, "train")
    dev_df = load_split(cfg.dataset, "dev")

    word_embedding_model = models.Transformer(cfg.model_name, max_seq_length=cfg.max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=cfg.device)

    train_examples = to_examples(train_df)
    dev_examples = to_examples(dev_df)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=cfg.batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_examples, batch_size=cfg.batch_size, name=f"{cfg.dataset}-dev"
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=cfg.epochs,
        warmup_steps=cfg.warmup_steps,
        evaluator=evaluator,
        evaluation_steps=max(10, len(train_dataloader) // 2),
        output_path=str(cfg.output_dir),
        use_amp=cfg.use_amp,
        optimizer_params={"lr": cfg.learning_rate},
        show_progress_bar=True,
    )

    # Final evaluation on dev with saved model
    best_model = SentenceTransformer(str(cfg.output_dir), device=cfg.device)
    dev_scores = []
    for _, row in dev_df.iterrows():
        emb = best_model.encode([row["sentence1"], row["sentence2"]], convert_to_tensor=True)
        score = util.cos_sim(emb[0], emb[1]).item() * 5.0
        dev_scores.append(score)
    metrics = compute_metrics(dev_df["label"].to_numpy(), np.array(dev_scores))
    metrics_path = cfg.output_dir / "dev_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Dev metrics ({cfg.dataset}): {metrics}")
    return metrics


def load_yaml_config(path: Path, preset: str) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    base = data.get("default", {})
    preset_conf = data.get(preset, {})
    merged = {**base, **preset_conf}
    merged["output_dir"] = REPO_ROOT / merged["output_dir"]
    return merged


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune SentenceTransformer on STS data.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--preset", type=str, default="default", help="YAML section to use (e.g., default, sick)")
    args = parser.parse_args()
    conf = load_yaml_config(Path(args.config), args.preset)
    return TrainConfig(
        model_name=conf["model_name"],
        epochs=int(conf["epochs"]),
        batch_size=int(conf["batch_size"]),
        learning_rate=float(conf["learning_rate"]),
        warmup_steps=int(conf["warmup_steps"]),
        max_seq_length=int(conf["max_seq_length"]),
        output_dir=conf["output_dir"],
        dataset=conf["dataset"],
        use_amp=bool(conf.get("use_amp", False)),
        device=str(conf.get("device", "cpu")),
    )


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
