"""
Entrypoint for Part 3.

Run from repo root:
    python -m scripts.run_part3

Notes:
- By default we cluster the training documents only (fast + enough for discovery).
- If you want to cluster train+test together, edit the docs list below.
"""

from __future__ import annotations
from src.labeling import make_cluster_label
from src.data import load_20newsgroups
from src.part3_clustering import run_part3


def main() -> None:
    X_train, y_train, X_test, y_test, _ = load_20newsgroups()

    # Recommended: cluster training docs for topic discovery.
    docs = X_train

    # Optional: cluster train+test together
    # docs = X_train + X_test

    run_part3(docs, use_llm_labels=False)


if __name__ == "__main__":
    main()
