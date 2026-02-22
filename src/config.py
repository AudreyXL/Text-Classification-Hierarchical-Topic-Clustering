"""
Central configuration for the project.

Keep all tunable values here so runs are reproducible and easy to explain.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# Reproducibility
RANDOM_STATE: int = 42

# Dataset
REMOVE_METADATA = ("headers", "footers", "quotes")

# Outputs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Part 1: shared vectorizer params
VECTORIZER_PARAMS = dict(
    max_features=30_000,
    stop_words="english",
    min_df=2,
    max_df=0.95,
)

# Part 1: model hyperparameters
PART1_MODELS = dict(
    multinomial_nb=dict(alpha=0.1),
    logistic_regression=dict(C=5.0, max_iter=2000, solver="lbfgs"),
    linear_svm=dict(C=1.0, max_iter=5000),
    random_forest=dict(n_estimators=300),
)

# Part 2: embeddings
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE: int = 64
NORMALIZE_EMBEDDINGS: bool = True

# Part 3: clustering
TOP_LEVEL_K_RANGE = range(2, 10)     # K must be < 10
REPRESENTATIVES_PER_CLUSTER: int = 5
SUBCLUSTERS_PER_BIG_CLUSTER: int = 3
