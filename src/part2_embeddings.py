"""
Part 2 — SentenceTransformer Embeddings + Classical Classifiers (Supervised)

We embed documents using a SentenceTransformer model, then train the same set of
classifiers as Part 1 on dense embeddings.

Outputs:
- outputs/part2_results.csv
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

import time
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from .config import (
    RANDOM_STATE,
    OUTPUT_DIR,
    EMBEDDING_MODEL_NAME,
    EMBED_BATCH_SIZE,
    NORMALIZE_EMBEDDINGS,
    PART1_MODELS,
)
from .metrics import compute_metrics, results_to_dataframe
from .utils import ensure_dir


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of documents into a dense matrix."""
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    emb = embedder.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    )
    return emb


def build_models() -> Dict[str, Any]:
    """
    Same model families as Part 1.

    Note: MultinomialNB is designed for non-negative count-like features.
    Dense embeddings are not counts and can include negatives, so NB is included
    only for completeness and usually performs poorly.
    """
    models = {
        "Multinomial NB (not ideal for embeddings)": MultinomialNB(**PART1_MODELS["multinomial_nb"]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                **PART1_MODELS["logistic_regression"],
                random_state=RANDOM_STATE,
            )),
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVC(
                **PART1_MODELS["linear_svm"],
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            **PART1_MODELS["random_forest"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return models


def run_part2(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Embed documents, train/evaluate models, and save a results CSV."""
    ensure_dir(OUTPUT_DIR)

    print("=" * 72)
    print("PART 2 — SentenceTransformer embeddings + classifiers")
    print("=" * 72)

    t0 = time.time()
    X_train_emb = embed_texts(X_train)
    X_test_emb = embed_texts(X_test)
    print(f"Embeddings: train={X_train_emb.shape}, test={X_test_emb.shape} ({time.time() - t0:.1f}s)")

    models = build_models()
    results: List[Dict[str, Any]] = []

    for name, model in models.items():
        t0 = time.time()

        # Make NB run by shifting embeddings to be non-negative.
        # This keeps the pipeline runnable, but NB is still not a great fit here.
        if isinstance(model, MultinomialNB):
            Xtr = X_train_emb - X_train_emb.min()
            Xte = X_test_emb - X_test_emb.min()
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
        else:
            model.fit(X_train_emb, y_train)
            y_pred = model.predict(X_test_emb)

        elapsed = time.time() - t0
        m = compute_metrics(y_test, y_pred)

        results.append({
            "Approach": "SentenceTransformer embeddings",
            "Model": name,
            "Accuracy": m["Accuracy"],
            "Macro-F1": m["Macro-F1"],
            "Time (s)": round(elapsed, 1),
            "y_pred": y_pred,
        })

        print(f"  {name:40s} Acc={m['Accuracy']:.4f} F1={m['Macro-F1']:.4f} ({elapsed:.1f}s)")

    df = results_to_dataframe(results)
    df = df.sort_values("Macro-F1", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    out_csv = OUTPUT_DIR / "part2_results.csv"
    df.to_csv(out_csv, index=True)
    print()
    print(f"Saved: {out_csv}")
    best = df.iloc[0]
    print(f"★ Best embeddings model: {best['Model']} (Acc={best['Accuracy']:.4f}, F1={best['Macro-F1']:.4f})")
    return df
