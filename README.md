# Ergatic Systems — Course Lab Work

Lab assignments for the "Ergatic Systems" course (2025). An ergatic system is a human-machine system where humans and automated components work together toward a shared goal.

## What's in here

This repo covers Task 2: exploratory analysis and dimensionality reduction on ECG signal data. The dataset contains ECG measurements (RR interval, P/QRS/T wave timings, axis values) along with free-text diagnostic reports and a `Healthy_Status` target label.

The notebook walks through:

- EDA with boxplots, correlation heatmaps, and class distribution
- NLP preprocessing — merging free-text report columns into a single field and encoding it via Word2Vec mean embeddings
- PCA to find the minimum number of components explaining 85%+ variance
- t-SNE visualization in 2D and 3D to inspect class separability

## Stack

Python 3, Jupyter Notebook, pandas, numpy, gensim, scikit-learn, seaborn, matplotlib

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/task2_ecg_analysis.ipynb
```

The notebook fetches the dataset directly from GitHub on first run — no manual download needed.

## Lab reports

PDF write-ups are in `reports/`. LaTeX source template (Springer Nature format) is in `latex-template/`.
