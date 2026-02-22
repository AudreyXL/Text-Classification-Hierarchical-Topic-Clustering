"""
Entrypoint for Part 2.

Run from repo root:
    python -m scripts.run_part2
"""

from __future__ import annotations

from src.data import load_20newsgroups
from src.part2_embeddings import run_part2


def main() -> None:
    X_train, y_train, X_test, y_test, _ = load_20newsgroups()
    run_part2(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
