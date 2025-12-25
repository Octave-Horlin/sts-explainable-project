import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baseline import build_features


def test_baseline_features_shape():
    df = pd.read_csv("data/processed/stsb_dev.csv").head(8)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    feats = build_features(
        np.asarray(model.encode(df["sentence1"].tolist())),
        np.asarray(model.encode(df["sentence2"].tolist())),
    )
    # shape = (n_samples, 4 * dim)
    dim = model.get_sentence_embedding_dimension()
    assert feats.shape == (len(df), dim * 4)


def test_advanced_forward():
    model_dir = Path("models/advanced_stsb")
    assert model_dir.exists(), "Train the advanced model first."
    model = SentenceTransformer(str(model_dir))
    s1 = "A cat sits on the mat."
    s2 = "A dog plays in the yard."
    emb = model.encode([s1, s2], convert_to_tensor=True)
    score = float((emb[0] @ emb[1]) / (emb[0].norm() * emb[1].norm()))
    assert -1.01 < score < 1.01
