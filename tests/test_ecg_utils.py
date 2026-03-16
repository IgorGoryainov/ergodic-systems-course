import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.ecg_utils import clean_reports, filter_outliers, pca_explained


# ---------------------------------------------------------------------------
# pca_explained
# ---------------------------------------------------------------------------


def _scaled(seed, n_samples=200, n_features=10):
    rng = np.random.RandomState(seed)
    return StandardScaler().fit_transform(rng.randn(n_samples, n_features))


def test_pca_explained_exceeds_threshold():
    X = _scaled(42)
    n, variance = pca_explained(X, threshold=0.8)
    assert variance > 0.8


def test_pca_explained_returns_at_least_two_components():
    X = _scaled(0)
    n, _ = pca_explained(X, threshold=0.1)
    assert n >= 2


def test_pca_explained_high_threshold():
    X = _scaled(1)
    n, variance = pca_explained(X, threshold=0.99)
    assert variance > 0.99
    assert n <= X.shape[1]


def test_pca_explained_caps_at_n_features():
    # Even with threshold=1.0, result cannot exceed feature count
    X = _scaled(7)
    n, variance = pca_explained(X, threshold=1.0)
    assert n == X.shape[1]
    assert variance == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# filter_outliers
# ---------------------------------------------------------------------------


def test_filter_outliers_removes_large_values():
    df = pd.DataFrame({"a": [100, 3000, 500], "b": [200, 100, 1999]})
    result = filter_outliers(df, ["a", "b"], threshold=2000)
    assert len(result) == 2
    assert 3000 not in result["a"].values


def test_filter_outliers_all_pass():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = filter_outliers(df, ["a", "b"], threshold=1000)
    assert len(result) == 3


def test_filter_outliers_ignores_missing_columns():
    df = pd.DataFrame({"a": [100, 200, 300]})
    result = filter_outliers(df, ["a", "nonexistent"], threshold=250)
    assert len(result) == 2


def test_filter_outliers_empty_column_list():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = filter_outliers(df, [], threshold=2)
    assert len(result) == 3


def test_filter_outliers_does_not_mutate_original():
    df = pd.DataFrame({"a": [100, 5000]})
    _ = filter_outliers(df, ["a"], threshold=200)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# clean_reports
# ---------------------------------------------------------------------------


def test_clean_reports_merges_columns():
    df = pd.DataFrame({"report_0": ["sinus", "af"], "report_1": ["normal", "flutter"], "other": [1, 2]})
    result = clean_reports(df, n_reports=2)
    assert "report" in result.columns
    assert "report_0" not in result.columns
    assert "report_1" not in result.columns
    assert "other" in result.columns
    assert result["report"].iloc[0] == "sinus normal"


def test_clean_reports_strips_nan_placeholders():
    df = pd.DataFrame({"report_0": ["sinus", "af"], "report_1": [np.nan, "flutter"]})
    result = clean_reports(df, n_reports=2)
    assert "nan" not in result["report"].iloc[0]
    assert result["report"].iloc[1] == "af flutter"


def test_clean_reports_does_not_mutate_original():
    df = pd.DataFrame({"report_0": ["x"], "report_1": ["y"]})
    _ = clean_reports(df, n_reports=2)
    assert "report_0" in df.columns


def test_clean_reports_handles_missing_report_columns():
    # Only report_0 exists; report_1 is absent — should not raise
    df = pd.DataFrame({"report_0": ["hello"], "value": [42]})
    result = clean_reports(df, n_reports=2)
    assert result["report"].iloc[0] == "hello"


def test_clean_reports_collapses_extra_whitespace():
    df = pd.DataFrame({"report_0": ["  word  "], "report_1": ["  other  "]})
    result = clean_reports(df, n_reports=2)
    assert "  " not in result["report"].iloc[0]
