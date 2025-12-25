# Semantic Textual Similarity with Explainability

## Overview
Semantic textual similarity (STS) system with explainable token alignments. We fine-tune sentence-transformer models on STS-B and SICK, compare against zero-shot/baseline variants, and provide a Streamlit demo with heatmaps for token-level similarities.

## Installation
Requirements: Python 3.10+, CPU is sufficient (GPU optional).
```bash
git clone <your-repo>
cd <your-project>
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
- STS-B (GLUE) and SICK.
- Preprocessed CSVs generated via:
```bash
python data/scripts/download_datasets.py
```
Outputs in `data/processed/`: `stsb_{train,dev}.csv`, `sick_{train,dev,test}.csv`, plus `dataset_stats.json`.

## Model Architecture
- Base encoder: `sentence-transformers/all-MiniLM-L6-v2`.
- Pooling: mean pooling.
- Loss: cosine similarity (scaled to 0-5).
- Baseline: frozen embeddings + Ridge on concat([u, v, |u-v|, u*v]).

## Training
Configs in `configs/config.yaml`.
```bash
# STS-B fine-tune (outputs models/advanced_stsb)
python src/training.py --config configs/config.yaml --preset default
# SICK fine-tune (outputs models/advanced_sick)
python src/training.py --config configs/config.yaml --preset sick
```

## Evaluation
Compare baseline vs tuned:
```bash
python src/evaluation.py --dataset stsb --advanced-dir models/advanced_stsb
python src/evaluation.py --dataset sick --advanced-dir models/advanced_sick --skip-baseline
```
Ablations zero-shot vs tuned:
```bash
python src/ablation.py --dataset stsb --tuned-dir models/advanced_stsb
python src/ablation.py --dataset sick --tuned-dir models/advanced_sick
```

## Demo
Streamlit app with token-level heatmaps:
```bash
streamlit run app.py
```

## Docker
Build and run the demo container:
```bash
docker build -t sts-demo .
docker run -p 8501:8501 sts-demo
```

## How to Run the Project (verification checklist)
Use a fresh shell in repo root:
```bash
# 0) Activate venv (if already created)
.\.venv\Scripts\activate

# 1) Install deps (if not already)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Generate data
python data/scripts/download_datasets.py

# 3) Baseline training/eval (STS-B)
python models/baseline.py --dataset stsb --device cpu

# 4) Fine-tune STS-B and SICK
python src/training.py --config configs/config.yaml --preset default
python src/training.py --config configs/config.yaml --preset sick

# 5) Evaluate and ablations
python src/evaluation.py --dataset stsb --advanced-dir models/advanced_stsb
python src/evaluation.py --dataset sick --advanced-dir models/advanced_sick --skip-baseline
python src/ablation.py --dataset stsb --tuned-dir models/advanced_stsb
python src/ablation.py --dataset sick --tuned-dir models/advanced_sick

# 6) Error analysis (threshold=1.0)
python src/error_analysis.py --dataset stsb --model models/advanced_stsb --split dev --threshold 1.0
python src/error_analysis.py --dataset sick --model models/advanced_sick --threshold 1.0

# 7) Explainability sample (heatmap)
python src/explain.py --model models/advanced_stsb --dataset stsb --index 0 --output-prefix outputs/stsb_sample0

# 8) Tests
pytest tests

# 9) Demo
streamlit run app.py
```
## Error Analysis
Collect high-error pairs and heuristic categories:
```bash
python src/error_analysis.py --dataset stsb --model models/advanced_stsb --split dev --threshold 1.0
python src/error_analysis.py --dataset sick --model models/advanced_sick --threshold 1.0
```
Outputs in `errors/`: `<dataset>_errors.csv`, `<dataset>_stats.json`.

## Project Structure
```
project/
├── data/
│   ├── processed/
│   └── scripts/download_datasets.py
├── models/
│   ├── artifacts/ (baseline weights)
│   ├── advanced_stsb/
│   └── advanced_sick/
├── src/
│   ├── training.py
│   ├── evaluation.py
│   ├── ablation.py
│   ├── explain.py
│   └── error_analysis.py
├── app.py
├── configs/config.yaml
├── requirements.txt
└── README.md
```

## Future Work
- Additional pooling/adapter variants for ablations.
- More granular error categories and saliency-based explanations.
- Dockerfile for reproducible deployment.

## References
- Reimers & Gurevych, 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Cer et al., 2017. SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Cross-lingual Focused Evaluation.
- Marelli et al., 2014. A SICK cure for the evaluation of compositional distributional semantic models.
