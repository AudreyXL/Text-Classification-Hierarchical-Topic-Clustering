"""
Entrypoint for Part 1.

Run from repo root:
    python -m scripts.run_part1
"""

from __future__ import annotations

from src.data import load_20newsgroups
from src.part1_supervised import run_part1

X_train, y_train, X_test, y_test, class_names = load_20newsgroups()

print("\nDataset Summary")
print("-" * 40)
print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
print(f"Number of classes: {len(class_names)}")
print()

run_part1(X_train, y_train, X_test, y_test)

