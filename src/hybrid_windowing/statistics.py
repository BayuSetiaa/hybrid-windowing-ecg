"""Statistical helpers for paired experiment comparisons."""

import json

import numpy as np
import pandas as pd
from scipy import stats


def load_results(results_dir):
    """Read per-run metrics, falling back to the aggregated table when runs are absent."""
    paths = sorted((results_dir / "runs").glob("*/metrics.json"))
    if paths:
        return pd.DataFrame([json.loads(path.read_text()) for path in paths])
    return pd.read_csv(results_dir / "results_all_runs.csv")


def confidence_interval(values, level=0.95):
    values = np.asarray(values, float)
    if len(values) < 2:
        return np.nan, np.nan
    margin = stats.t.ppf(0.5 + level / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean() - margin, values.mean() + margin


def holm_adjust(p_values):
    """Return Holm-adjusted p-values."""
    p_values = np.asarray(p_values, float)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(np.argsort(p_values)):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def mcnemar_exact(labels, predictions_a, predictions_b):
    """Return the discordant counts and the exact McNemar p-value."""
    correct_a, correct_b = predictions_a == labels, predictions_b == labels
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    total = only_a + only_b
    p_value = stats.binomtest(min(only_a, only_b), total, 0.5).pvalue if total > 0 else 1.0
    return only_a, only_b, p_value
