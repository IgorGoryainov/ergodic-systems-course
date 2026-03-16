"""Utility functions for ECG data preprocessing and dimensionality reduction."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def pca_explained(X, threshold):
    """Find the minimum number of PCA components that exceed a variance threshold.

    Args:
        X: Scaled feature matrix of shape (n_samples, n_features).
        threshold: Cumulative explained variance target, e.g. 0.85.

    Returns:
        Tuple of (n_components, cumulative_variance) where n_components is the
        smallest number of components (starting from 2) that exceed threshold.
        Returns (n_features, 1.0) if no stopping point is found.
    """
    n_features = X.shape[1]
    for n in range(2, n_features + 1):
        pca = PCA(n_components=n).fit(X)
        variance = sum(pca.explained_variance_ratio_)
        if variance > threshold:
            return n, variance
    return n_features, 1.0


def filter_outliers(df, columns, threshold=2000):
    """Remove rows where any value in the specified columns exceeds a threshold.

    Args:
        df: Input DataFrame.
        columns: Column names to check. Columns not present in df are ignored.
        threshold: Rows with any value >= threshold in these columns are dropped.

    Returns:
        Filtered DataFrame (copy, original is unchanged).
    """
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return df.copy()
    return df[(df[valid_cols] < threshold).all(axis=1)].copy()


def clean_reports(df, n_reports=18):
    """Merge individual report_N columns into a single 'report' text column.

    Concatenates report_0 through report_(n_reports-1) into one space-separated
    string per row, then strips 'nan' placeholders and extra whitespace.

    Args:
        df: DataFrame containing report_0 ... report_(n_reports-1) columns.
        n_reports: Number of report columns to merge.

    Returns:
        DataFrame (copy) with a merged 'report' column. Original report_N
        columns are dropped.
    """
    report_cols = [f"report_{i}" for i in range(n_reports)]
    present = [col for col in report_cols if col in df.columns]

    result = df.copy()
    # Use apply + explicit str() to stay compatible with pandas 1.x and 2.x.
    # DataFrame.agg(" ".join, axis=1) receives raw floats for NaN cells in
    # pandas 2.x even after .astype(str), causing a TypeError.
    result["report"] = (
        result[present]
        .apply(lambda row: " ".join(str(v) for v in row), axis=1)
        .str.replace(r"\bnan\b", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    result = result.drop(columns=present)
    return result


def get_sentence_embedding(sentence, model):
    """Compute the mean Word2Vec embedding for a sentence.

    Words not present in the model vocabulary are skipped. Returns a zero
    vector if no words are found in the vocabulary.

    Args:
        sentence: Input string.
        model: Trained gensim Word2Vec model.

    Returns:
        1-D numpy array of shape (vector_size,).
    """
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)
