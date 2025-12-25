import pandas as pd
from pathlib import Path


def test_processed_files_exist():
    base = Path("data/processed")
    required = [
        "stsb_train.csv",
        "stsb_dev.csv",
        "sick_train.csv",
        "sick_dev.csv",
        "sick_test.csv",
        "dataset_stats.json",
    ]
    for name in required:
        assert (base / name).exists(), f"Missing file: {name}"


def test_label_range_stsb():
    df = pd.read_csv("data/processed/stsb_train.csv")
    assert df["label"].between(0, 5).all()


def test_label_range_sick():
    df = pd.read_csv("data/processed/sick_train.csv")
    assert df["label"].between(0, 5).all()
