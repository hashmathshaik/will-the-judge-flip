from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def score(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    result: dict[str, float | None] = {
        "pfr": float(y.mean()) if len(y) else None,
        "auroc": None,
        "auprc": None,
        "brier": float(np.mean((probabilities - y) ** 2)) if len(y) else None,
    }
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, probabilities))
        result["auprc"] = float(average_precision_score(y, probabilities))
    return result


def score_no_signal(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prevalence = float(y.mean()) if len(y) else None
    estimable = len(np.unique(y)) == 2
    return {
        "pfr": prevalence,
        "auroc": 0.5 if estimable else None,
        "auprc": prevalence if estimable else None,
        "brier": float(np.mean((probabilities - y) ** 2)) if len(y) else None,
    }


def _bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def bootstrap_ci(
    y: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    repetitions: int = 2000,
    seed: int = 20260828,
) -> dict[str, dict[str, float | int | None]]:
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(repetitions):
        indices = _bootstrap_indices(groups, rng)
        values = score(y[indices], probabilities[indices])
        for metric in samples:
            if values[metric] is not None:
                samples[metric].append(float(values[metric]))
    result: dict[str, dict[str, float | int | None]] = {}
    for metric, values in samples.items():
        if not values:
            result[metric] = {"lower": None, "upper": None, "valid_repetitions": 0}
        else:
            lower, upper = np.percentile(values, [2.5, 97.5])
            result[metric] = {
                "lower": float(lower),
                "upper": float(upper),
                "valid_repetitions": len(values),
            }
    return result


def paired_difference_ci(
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    groups: np.ndarray,
    repetitions: int = 2000,
    seed: int = 20260828,
) -> dict[str, dict[str, float | int | None]]:
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(repetitions):
        indices = _bootstrap_indices(groups, rng)
        first_score = score(y[indices], first[indices])
        second_score = score(y[indices], second[indices])
        for metric in samples:
            if first_score[metric] is not None and second_score[metric] is not None:
                samples[metric].append(float(first_score[metric] - second_score[metric]))
    result: dict[str, dict[str, float | int | None]] = {}
    for metric, values in samples.items():
        if not values:
            result[metric] = {"lower": None, "upper": None, "valid_repetitions": 0}
        else:
            lower, upper = np.percentile(values, [2.5, 97.5])
            result[metric] = {
                "lower": float(lower),
                "upper": float(upper),
                "valid_repetitions": len(values),
            }
    return result


def _summarize_samples(
    samples: dict[str, list[float]],
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for metric, values in samples.items():
        if not values:
            result[metric] = {"lower": None, "upper": None, "valid_repetitions": 0}
        else:
            lower, upper = np.percentile(values, [2.5, 97.5])
            result[metric] = {
                "lower": float(lower),
                "upper": float(upper),
                "valid_repetitions": len(values),
            }
    return result


def bootstrap_prediction_suite(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    reference_name: str,
    no_signal_names: set[str] | None = None,
    repetitions: int = 2000,
    seed: int = 20260828,
) -> dict[str, dict[str, dict[str, dict[str, float | int | None]]]]:
    if reference_name not in predictions:
        raise ValueError(f"Unknown reference predictor: {reference_name}")
    no_signal_names = no_signal_names or set()
    metrics = ("auroc", "auprc", "brier")
    predictor_samples = {name: {metric: [] for metric in metrics} for name in predictions}
    paired_samples = {
        name: {metric: [] for metric in metrics} for name in predictions if name != reference_name
    }
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        indices = _bootstrap_indices(groups, rng)
        replicate_scores = {
            name: (
                score_no_signal(y[indices], values[indices])
                if name in no_signal_names
                else score(y[indices], values[indices])
            )
            for name, values in predictions.items()
        }
        reference = replicate_scores[reference_name]
        for name, values in replicate_scores.items():
            for metric in metrics:
                if values[metric] is not None:
                    predictor_samples[name][metric].append(float(values[metric]))
                if (
                    name != reference_name
                    and values[metric] is not None
                    and reference[metric] is not None
                ):
                    paired_samples[name][metric].append(float(reference[metric] - values[metric]))
    return {
        "predictors": {
            name: _summarize_samples(samples) for name, samples in predictor_samples.items()
        },
        "reference_minus_predictor": {
            name: _summarize_samples(samples) for name, samples in paired_samples.items()
        },
    }


def stratified_auroc(y: np.ndarray, probabilities: np.ndarray, strata: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    strata = np.asarray(strata)
    numerator = 0.0
    denominator = 0.0
    for level in np.unique(strata):
        mask = strata == level
        labels = y[mask]
        positives = int(labels.sum())
        negatives = int(len(labels) - positives)
        if positives == 0 or negatives == 0:
            continue
        weight = float(positives * negatives)
        numerator += float(roc_auc_score(labels, probabilities[mask])) * weight
        denominator += weight
    if denominator == 0.0:
        return None
    return numerator / denominator


def bootstrap_stratified_suite(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    strata: np.ndarray,
    reference_name: str,
    repetitions: int = 2000,
    seed: int = 20260828,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    if reference_name not in predictions:
        raise ValueError(f"Unknown reference predictor: {reference_name}")
    predictor_samples: dict[str, list[float]] = {name: [] for name in predictions}
    paired_samples: dict[str, list[float]] = {
        name: [] for name in predictions if name != reference_name
    }
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        indices = _bootstrap_indices(groups, rng)
        replicate = {
            name: stratified_auroc(y[indices], values[indices], strata[indices])
            for name, values in predictions.items()
        }
        reference = replicate[reference_name]
        for name, value in replicate.items():
            if value is not None:
                predictor_samples[name].append(value)
            if name != reference_name and value is not None and reference is not None:
                paired_samples[name].append(reference - value)

    def summarize(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"lower": None, "upper": None, "valid_repetitions": 0}
        lower, upper = np.percentile(values, [2.5, 97.5])
        return {
            "lower": float(lower),
            "upper": float(upper),
            "valid_repetitions": len(values),
        }

    return {
        "predictors": {name: summarize(values) for name, values in predictor_samples.items()},
        "reference_minus_predictor": {
            name: summarize(values) for name, values in paired_samples.items()
        },
    }
