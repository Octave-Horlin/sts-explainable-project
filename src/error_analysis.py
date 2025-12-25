"""
Collect errors (large prediction gaps) and categorize them for STS.

Outputs:
- errors/<dataset>_errors.csv with columns: sentence1, sentence2, gold, pred, abs_error, category
- errors/<dataset>_stats.json with counts per category
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
ERROR_DIR = REPO_ROOT / "errors"


def load_split(dataset: str, split: str) -> pd.DataFrame:
    path = DATA_DIR / f"{dataset}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")
    return pd.read_csv(path)


def heuristic_category(row: pd.Series) -> str:
    s1 = str(row["sentence1"]).lower()
    s2 = str(row["sentence2"]).lower()
    if any(tok in s1 and tok not in s2 for tok in ["not", "no", "never"]) or any(
        tok in s2 and tok not in s1 for tok in ["not", "no", "never"]
    ):
        return "negation_mismatch"
    if len(s1.split()) > 20 or len(s2.split()) > 20:
        return "long_sentence"
    if len(set(s1.split()) & set(s2.split())) <= 2:
        return "low_overlap"
    return "other"


def collect_errors(dataset: str, model_path: Path, split: str, threshold: float) -> pd.DataFrame:
    df = load_split(dataset, split)
    model = SentenceTransformer(str(model_path))
    preds = []
    for _, row in df.iterrows():
        emb = model.encode([row["sentence1"], row["sentence2"]], convert_to_tensor=True)
        score = util.cos_sim(emb[0], emb[1]).item() * 5.0
        preds.append(score)
    df["pred"] = preds
    df["abs_error"] = np.abs(df["pred"] - df["label"])
    errors = df[df["abs_error"] >= threshold].copy()
    errors["category"] = errors.apply(heuristic_category, axis=1)
    return errors


def stats_by_category(errors: pd.DataFrame) -> Dict[str, int]:
    counts = errors["category"].value_counts().to_dict()
    return {k: int(v) for k, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and categorize STS errors.")
    parser.add_argument("--dataset", choices=["stsb", "sick"], required=True)
    parser.add_argument("--model", type=str, default="models/advanced_stsb")
    parser.add_argument("--split", type=str, default="dev", help="dev for stsb, test for sick")
    parser.add_argument("--threshold", type=float, default=1.0, help="Absolute error threshold to log")
    args = parser.parse_args()

    if args.dataset == "sick" and args.split == "dev":
        args.split = "test"  # SICK uses test for evaluation

    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    errors = collect_errors(args.dataset, Path(args.model), args.split, args.threshold)
    out_csv = ERROR_DIR / f"{args.dataset}_errors.csv"
    out_stats = ERROR_DIR / f"{args.dataset}_stats.json"
    errors.to_csv(out_csv, index=False)
    stats = stats_by_category(errors)
    out_stats.write_text(json.dumps(stats, indent=2))
    print(f"Saved errors to {out_csv} (count={len(errors)})")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
