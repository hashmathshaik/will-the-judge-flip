from __future__ import annotations

import hashlib

import numpy as np

from position_bias.metrics import _bootstrap_indices


def rank_pairs(risk: np.ndarray, pair_ids: list[str]) -> np.ndarray:
    risk = np.asarray(risk, dtype=float)
    if risk.shape != (len(pair_ids),) or not np.isfinite(risk).all():
        raise ValueError("Risk must be a finite score for each pair")
    hashes = [hashlib.sha256(str(pair).encode()).hexdigest() for pair in pair_ids]
    return np.asarray(sorted(range(len(risk)), key=lambda i: (-risk[i], hashes[i])))


def check_budget(y: np.ndarray, risk: np.ndarray, pair_ids: list[str], budget: float) -> dict:
    y = np.asarray(y, dtype=int)
    if not 0 <= budget <= 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Budget must lie in [0,1] and labels must be nonempty binary values")
    if len(y) != len(pair_ids):
        raise ValueError("Pair IDs and labels must align")
    count = round(budget * len(y))
    selected = rank_pairs(risk, pair_ids)[:count]
    detected = int(y[selected].sum())
    total = int(y.sum())
    return {
        "budget": budget,
        "n": len(y),
        "checked": count,
        "reversals": total,
        "detected": detected,
        "recall": detected / total if total else None,
        "precision": detected / count if count else None,
        "unchecked": len(y) - count,
        "missed": total - detected,
        "unchecked_reversal_rate": (total - detected) / (len(y) - count)
        if count < len(y)
        else None,
        "judgment_calls": len(y) + count,
        "call_savings_fraction": (len(y) - count) / (2 * len(y)),
    }


def recall_crossing(
    y: np.ndarray, risk: np.ndarray, pair_ids: list[str], recall: float = 0.9
) -> dict | None:
    if not 0 < recall <= 1:
        raise ValueError("Recall target must lie in (0,1]")
    if not np.asarray(y).sum():
        return None
    cumulative = np.cumsum(np.asarray(y)[rank_pairs(risk, pair_ids)])
    for percent in range(101):
        count = round(percent / 100 * len(y))
        detected = int(cumulative[count - 1]) if count else 0
        if detected / np.asarray(y).sum() >= recall:
            return check_budget(y, risk, pair_ids, percent / 100)
    raise RuntimeError("Exhaustive checking failed to detect every reversal")


def budget_interval(
    y: np.ndarray,
    risk: np.ndarray,
    pair_ids: list[str],
    groups: np.ndarray,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        indices = _bootstrap_indices(groups, rng)
        crossing = recall_crossing(y[indices], risk[indices], [pair_ids[i] for i in indices])
        if crossing:
            samples.append(crossing["budget"])
    bounds = np.percentile(samples, [2.5, 97.5]).tolist() if samples else [None, None]
    return {
        "lower": bounds[0],
        "upper": bounds[1],
        "valid_repetitions": len(samples),
        "predictors_fixed": True,
    }
