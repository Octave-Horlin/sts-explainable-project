"""
Streamlit demo for STS with explainability.

Run:
  streamlit run app.py
"""

import io
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util

REPO_ROOT = Path(__file__).resolve().parent
MODELS = {
    "STS-B (tuned)": REPO_ROOT / "models" / "advanced_stsb",
    "SICK (tuned)": REPO_ROOT / "models" / "advanced_sick",
    "Zero-shot (all-MiniLM-L6-v2)": "sentence-transformers/all-MiniLM-L6-v2",
}


@st.cache_resource
def load_model(path: str):
    return SentenceTransformer(path)


def predict(model: SentenceTransformer, s1: str, s2: str) -> float:
    emb = model.encode([s1, s2], convert_to_tensor=True)
    score = util.cos_sim(emb[0], emb[1]).item() * 5.0
    return score


def token_heatmap(model: SentenceTransformer, text1: str, text2: str):
    tokens = model.tokenize([text1, text2])
    with torch.no_grad():
        output = model(tokens)
    emb1 = output["token_embeddings"][0][tokens["attention_mask"][0].bool()]
    emb2 = output["token_embeddings"][1][tokens["attention_mask"][1].bool()]
    tok1 = model.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0][tokens["attention_mask"][0].bool()].tolist())
    tok2 = model.tokenizer.convert_ids_to_tokens(tokens["input_ids"][1][tokens["attention_mask"][1].bool()].tolist())
    sim = util.pytorch_cos_sim(emb1, emb2).cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(6, len(tok2) * 0.4), max(4, len(tok1) * 0.4)))
    im = ax.imshow(sim, cmap="viridis")
    ax.set_xticks(range(len(tok2)))
    ax.set_yticks(range(len(tok1)))
    ax.set_xticklabels(tok2, rotation=45, ha="right")
    ax.set_yticklabels(tok1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    return buf.getvalue()


def main():
    st.title("Semantic Textual Similarity Demo")
    st.markdown("Compute similarity scores (0-5) and visualize token alignments.")

    model_choice = st.selectbox("Model", list(MODELS.keys()))
    model_path = MODELS[model_choice]
    model = load_model(str(model_path))

    examples: List[Tuple[str, str]] = [
        ("A man with a hard hat is dancing.", "A man wearing a hard hat is dancing."),
        ("A woman is playing violin.", "A person is performing music with a violin."),
        ("The cat sits on the mat.", "A dog plays in the yard."),
    ]
    ex_idx = st.selectbox("Example pair", range(len(examples)), format_func=lambda i: examples[i][0])
    default1, default2 = examples[ex_idx]

    s1 = st.text_area("Sentence 1", default1, height=120)
    s2 = st.text_area("Sentence 2", default2, height=120)

    if st.button("Compute similarity"):
        start = time.time()
        score = predict(model, s1, s2)
        elapsed = (time.time() - start) * 1000
        st.metric("Similarity (0-5)", f"{score:.3f}", help="Scaled cosine similarity")
        st.caption(f"Inference time: {elapsed:.1f} ms")

        st.subheader("Token alignment heatmap")
        with st.spinner("Computing token similarities..."):
            img = token_heatmap(model, s1, s2)
        st.image(img)


if __name__ == "__main__":
    main()
