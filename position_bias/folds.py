from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def make_grouped_folds(rows: list[dict], n_splits: int = 5, seed: int = 42) -> list[int]:
    groups = np.asarray([str(row["group_id"]) for row in rows])
    if n_splits < 2 or len(set(groups)) < n_splits:
        raise ValueError(f"Need at least {n_splits} distinct question groups")
    assignments = np.full(len(rows), -1, dtype=int)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, held_out) in enumerate(splitter.split(np.zeros(len(rows)), groups=groups)):
        assignments[held_out] = fold
    return assignments.tolist()
