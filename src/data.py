"""
Dataset loading utilities.

We use the canonical 20 Newsgroups train/test split and remove headers/footers/quotes
to reduce metadata leakage.
"""

from __future__ import annotations
from typing import List, Tuple

from sklearn.datasets import fetch_20newsgroups

from .config import REMOVE_METADATA, RANDOM_STATE


def load_20newsgroups() -> Tuple[List[str], List[int], List[str], List[int], List[str]]:
    """
    Returns:
        X_train, y_train, X_test, y_test, class_names
    """
    train = fetch_20newsgroups(
        subset="train",
        remove=REMOVE_METADATA,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    test = fetch_20newsgroups(
        subset="test",
        remove=REMOVE_METADATA,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    X_train, y_train = train.data, train.target
    X_test, y_test = test.data, test.target
    class_names = train.target_names
    return X_train, y_train, X_test, y_test, class_names
    
