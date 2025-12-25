"""
Download and preprocess STS-B (GLUE) and SICK relatedness datasets.

Outputs (relative to repo root):
- data/processed/stsb_train.csv
- data/processed/stsb_dev.csv
- data/processed/stsb_test.csv (labels absent in GLUE test -> skipped)
- data/processed/sick_train.csv
- data/processed/sick_dev.csv
- data/processed/sick_test.csv
- data/processed/dataset_stats.json
"""

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import pandas as pd
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def save_split(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def compute_stats(df: pd.DataFrame, label_col: str) -> Dict[str, float]:
    labels = df[label_col].dropna()
    return {
        "count": int(labels.shape[0]),
        "mean": float(labels.mean()),
        "std": float(labels.std()),
        "min": float(labels.min()),
        "max": float(labels.max()),
    }


def prepare_stsb() -> Dict[str, Dict[str, float]]:
    """Load GLUE STS-B and write train/dev splits (test has no labels)."""
    ds = load_dataset("glue", "stsb")
    stats: Dict[str, Dict[str, float]] = {}

    for split in ("train", "validation"):
        frame = ds[split].to_pandas()[["sentence1", "sentence2", "label"]]
        save_split(frame, PROCESSED_DIR / f"stsb_{'dev' if split == 'validation' else 'train'}.csv")
        stats[f"stsb_{'dev' if split == 'validation' else 'train'}"] = compute_stats(frame, "label")

    # Skip test labels (not provided in GLUE).
    return stats


def prepare_sick() -> Dict[str, Dict[str, float]]:
    """Load SICK relatedness and write train/dev/test, scaling to 0-5."""
    stats: Dict[str, Dict[str, float]] = {}
    sick_dir = REPO_ROOT / "data" / "raw" / "SICK"
    sick_dir.mkdir(parents=True, exist_ok=True)
    sick_txt = sick_dir / "SICK.txt"

    if not sick_txt.exists():
        zip_path = sick_dir / "SICK.zip"
        url = "https://zenodo.org/record/2787612/files/SICK.zip?download=1"
        print(f"Downloading SICK from {url}")
        urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(sick_dir)

    frame_all = pd.read_csv(sick_txt, sep="\t")
    frame_all = frame_all.rename(columns={"sentence_A": "sentence1", "sentence_B": "sentence2"})

    split_map = {
        "TRAIN": "train",
        "TRIAL": "dev",
        "TEST": "test",
    }

    for semeval_label, split_name in split_map.items():
        frame = frame_all[frame_all["SemEval_set"] == semeval_label].copy()
        # Shift [1,5] -> [0,4] then scale to [0,5] for consistency with STS-B.
        frame["label"] = (frame["relatedness_score"].astype(float) - 1.0) * (5.0 / 4.0)
        frame = frame[["sentence1", "sentence2", "label"]]

        save_split(frame, PROCESSED_DIR / f"sick_{split_name}.csv")
        stats[f"sick_{split_name}"] = compute_stats(frame, "label")

    return stats


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    stats = {}
    stats.update(prepare_stsb())
    stats.update(prepare_sick())

    stats_path = PROCESSED_DIR / "dataset_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote stats to {stats_path}")


if __name__ == "__main__":
    main()
