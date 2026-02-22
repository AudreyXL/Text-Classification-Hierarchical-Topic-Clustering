"""
Part 1 — TF-IDF / BoW + Classical Classifiers (Supervised)

This runs 8 pipelines:
- 2 vectorizers (BoW, TF-IDF)
- 4 classifiers (MNB, Logistic Regression, Linear SVM, Random Forest)

Outputs:
- outputs/part1_results.csv
- outputs/part1_model_comparison.png
"""

from __future__ import annotations
from typing import Dict, Any, List

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from .config import RANDOM_STATE, VECTORIZER_PARAMS, PART1_MODELS, OUTPUT_DIR
from .metrics import compute_metrics, results_to_dataframe
from .utils import ensure_dir


def build_pipelines() -> Dict[str, Pipeline]:
    """Create all (vectorizer, classifier) combinations as sklearn Pipelines."""
    vectorizers = {
        "BoW": CountVectorizer(**VECTORIZER_PARAMS),
        "TF-IDF": TfidfVectorizer(**VECTORIZER_PARAMS, sublinear_tf=True),
    }

    classifiers = {
        "Multinomial NB": MultinomialNB(**PART1_MODELS["multinomial_nb"]),
        "Logistic Regression": LogisticRegression(
            **PART1_MODELS["logistic_regression"],
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(
            **PART1_MODELS["linear_svm"],
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            **PART1_MODELS["random_forest"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    pipelines: Dict[str, Pipeline] = {}
    for vec_name, vec in vectorizers.items():
        for clf_name, clf in classifiers.items():
            name = f"{vec_name} + {clf_name}"
            pipelines[name] = Pipeline([("vectorizer", clone(vec)), ("model", clone(clf))])
    return pipelines


def run_part1(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Train and evaluate all Part 1 pipelines.
    Returns a sorted results DataFrame.
    """
    ensure_dir(OUTPUT_DIR)
    pipelines = build_pipelines()
    results: List[Dict[str, Any]] = []

    print("=" * 72)
    print("PART 1 — Training & evaluating 8 pipelines")
    print("=" * 72)

    for name, pipe in pipelines.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)          # train on train only
        y_pred = pipe.predict(X_test)       # evaluate on test
        elapsed = time.time() - t0

        vec_name, clf_name = name.split(" + ", 1)
        m = compute_metrics(y_test, y_pred)

        results.append({
            "Pipeline": name,
            "Vectorizer": vec_name,
            "Classifier": clf_name,
            "Accuracy": m["Accuracy"],
            "Macro-F1": m["Macro-F1"],
            "Time (s)": round(elapsed, 1),
            "y_pred": y_pred,
        })

        print(f"  {name:35s} Acc={m['Accuracy']:.4f} F1={m['Macro-F1']:.4f} ({elapsed:.1f}s)")

    df = results_to_dataframe(results)
    df = df.sort_values("Macro-F1", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    out_csv = OUTPUT_DIR / "part1_results.csv"
    df.to_csv(out_csv, index=True)

    # Plot: BoW vs TF-IDF per classifier (Macro-F1)
    plot_bow_vs_tfidf(df, OUTPUT_DIR / "part1_model_comparison.png")

    best = df.iloc[0]
    print()
    print(f"★ Best pipeline: {best['Pipeline']} (Acc={best['Accuracy']:.4f}, F1={best['Macro-F1']:.4f})")
    print(f"Saved: {out_csv}")
    return df


def plot_bow_vs_tfidf(df: pd.DataFrame, out_path) -> None:
    """
    Create a grouped bar chart: BoW vs TF-IDF Macro-F1 per classifier.

    We do NOT hard-code colors (keeps it simple and consistent).
    """
    # Ensure consistent order
    clf_labels = list(df["Classifier"].unique())

    def get_f1(vec_name: str, clf_name: str) -> float:
        row = df[(df["Vectorizer"] == vec_name) & (df["Classifier"] == clf_name)]
        return float(row["Macro-F1"].iloc[0])

    bow_scores = [get_f1("BoW", c) for c in clf_labels]
    tfidf_scores = [get_f1("TF-IDF", c) for c in clf_labels]

    x = np.arange(len(clf_labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, bow_scores, w, label="BoW")
    b2 = ax.bar(x + w/2, tfidf_scores, w, label="TF-IDF")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Macro-F1")
    ax.set_title("BoW vs TF-IDF — Macro-F1 by Classifier")
    ax.set_xticks(x)
    ax.set_xticklabels(clf_labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
