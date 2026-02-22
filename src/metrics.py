"""
Metrics helpers.

We keep this small and reusable across parts.
"""

from __future__ import annotations
from typing import Dict, Any, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute the required evaluation metrics."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro")),
    }


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of dicts into a clean DataFrame (drops raw predictions if present)."""
    df = pd.DataFrame(results)
    if "y_pred" in df.columns:
        df = df.drop(columns=["y_pred"])
    return df
