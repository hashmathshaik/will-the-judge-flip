from __future__ import annotations

import warnings
from collections import Counter
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def flip_burden_bin(number_of_flips: int) -> str:
    if number_of_flips <= 1:
        return "low"
    if number_of_flips == 2:
        return "medium"
    return "high"


def make_grouped_folds(
    rows: list[dict[str, Any]],
    n_splits: int = 5,
    seed: int = 20260828,
) -> list[int]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    groups = np.asarray([str(row["group_id"]) for row in rows])
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"Need {n_splits} groups but only found {len(np.unique(groups))}")
    strata = np.asarray([f"{row['coarse_domain']}::{row['flip_burden_bin']}" for row in rows])
    stratum_counts = Counter(strata.tolist())
    if max(stratum_counts.values()) < n_splits:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
    assignments = np.full(len(rows), -1, dtype=np.int64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class in y has only")
        for fold, (_, test_indices) in enumerate(
            splitter.split(np.zeros(len(rows)), strata, groups)
        ):
            assignments[test_indices] = fold
    if np.any(assignments < 0) or len(set(assignments.tolist())) != n_splits:
        raise RuntimeError("Grouped fold assignment did not populate every fold")
    for group in np.unique(groups):
        if len(set(assignments[groups == group].tolist())) != 1:
            raise RuntimeError(f"Group was split across folds: {group}")
    return assignments.tolist()


def add_flip_burden(rows: list[dict[str, Any]], labels: list[int]) -> list[dict[str, Any]]:
    if len(rows) != len(labels):
        raise ValueError("Rows and labels have different lengths")
    return [
        {
            **row,
            "flip_burden": int(label),
            "flip_burden_bin": flip_burden_bin(int(label)),
        }
        for row, label in zip(rows, labels)
    ]
