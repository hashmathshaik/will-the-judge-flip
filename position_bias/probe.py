from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.asarray(x, dtype=np.float64).mean(axis=0)
        scale = np.asarray(x, dtype=np.float64).std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale


@dataclass
class LogisticModel:
    weights: np.ndarray | None
    intercept: float
    constant: float | None
    standardizer: Standardizer
    lambda_value: float
    converged: bool

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(x), self.constant, dtype=np.float64)
        if self.weights is None:
            raise RuntimeError("A non-constant model must have fitted weights")
        z = self.standardizer.transform(x) @ self.weights + self.intercept
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(x: np.ndarray, y: np.ndarray, lambda_value: float) -> LogisticModel:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    standardizer = Standardizer.fit(x)
    x_scaled = standardizer.transform(x)
    if np.unique(y).size < 2:
        prevalence = float(y.mean())
        return LogisticModel(None, 0.0, prevalence, standardizer, lambda_value, True)

    dimension = x_scaled.shape[1]
    initial = np.zeros(dimension + 1, dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights, intercept = parameters[:-1], parameters[-1]
        logits = x_scaled @ weights + intercept
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        loss += lambda_value * float(weights @ weights) + 1e-8 * intercept**2
        probabilities = 1.0 / (1.0 + np.exp(np.clip(-logits, -60.0, 60.0)))
        gradient = np.empty_like(parameters)
        gradient[:-1] = (x_scaled.T @ (probabilities - y)) / len(y)
        gradient[:-1] += 2.0 * lambda_value * weights
        gradient[-1] = float(np.mean(probabilities - y)) + 2e-8 * intercept
        return float(loss), gradient

    result = minimize(
        lambda parameters: objective(parameters),
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("Logistic solver returned non-finite parameters")
    return LogisticModel(
        weights=result.x[:-1],
        intercept=float(result.x[-1]),
        constant=None,
        standardizer=standardizer,
        lambda_value=lambda_value,
        converged=bool(result.success),
    )


def brier_score(y: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(y)) ** 2))


def select_lambda(
    x: np.ndarray,
    y: np.ndarray,
    inner_folds: list[int],
    lambdas: tuple[float, ...] = LAMBDA_GRID,
) -> float:
    scores: list[tuple[float, float]] = []
    for lambda_value in lambdas:
        validation_scores: list[float] = []
        for fold in sorted(set(inner_folds)):
            train = np.asarray([i for i, value in enumerate(inner_folds) if value != fold])
            validation = np.asarray([i for i, value in enumerate(inner_folds) if value == fold])
            model = fit_logistic(x[train], y[train], lambda_value)
            validation_scores.append(brier_score(y[validation], model.predict(x[validation])))
        scores.append((float(np.mean(validation_scores)), lambda_value))
    return min(scores, key=lambda pair: (pair[0], pair[1]))[1]


def nested_predict(
    features: np.ndarray,
    y: np.ndarray,
    outer_folds: list[int],
    inner_fold_builder: Callable[[list[dict[str, Any]], int], list[int]],
    row_metadata: list[dict[str, Any]],
    layer_numbers: list[int] | None = None,
    evaluation_features: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = np.asarray(features)
    if evaluation_features is not None:
        evaluation_features = np.asarray(evaluation_features)
        if evaluation_features.shape != features.shape:
            raise ValueError("Evaluation features must match the training feature shape")
    y = np.asarray(y, dtype=np.int64)
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    choices: list[dict[str, Any]] = []
    candidate_layers = list(range(features.shape[1]))

    for outer_fold in sorted(set(outer_folds)):
        train_indices = [i for i, fold in enumerate(outer_folds) if fold != outer_fold]
        test_indices = [i for i, fold in enumerate(outer_folds) if fold == outer_fold]
        train_rows = [row_metadata[i] for i in train_indices]
        inner_folds = inner_fold_builder(train_rows, outer_fold)
        candidates: list[tuple[float, int, float]] = []
        for layer_index in candidate_layers:
            layer_x = features[:, layer_index, :] if features.ndim == 3 else features
            for lambda_value in LAMBDA_GRID:
                validation_scores: list[float] = []
                for inner_fold in sorted(set(inner_folds)):
                    local_train = np.asarray(
                        [j for j, value in enumerate(inner_folds) if value != inner_fold]
                    )
                    local_validation = np.asarray(
                        [j for j, value in enumerate(inner_folds) if value == inner_fold]
                    )
                    model = fit_logistic(
                        layer_x[np.asarray(train_indices)[local_train]],
                        y[np.asarray(train_indices)[local_train]],
                        lambda_value,
                    )
                    validation_scores.append(
                        brier_score(
                            y[np.asarray(train_indices)[local_validation]],
                            model.predict(layer_x[np.asarray(train_indices)[local_validation]]),
                        )
                    )
                candidates.append((float(np.mean(validation_scores)), layer_index, lambda_value))
        best_score, best_layer_index, best_lambda = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        layer_x = features[:, best_layer_index, :] if features.ndim == 3 else features
        model = fit_logistic(layer_x[train_indices], y[train_indices], best_lambda)
        if evaluation_features is None:
            layer_eval = layer_x
        else:
            layer_eval = (
                evaluation_features[:, best_layer_index, :]
                if evaluation_features.ndim == 3
                else evaluation_features
            )
        predictions[test_indices] = model.predict(layer_eval[test_indices])
        choices.append(
            {
                "outer_fold": outer_fold,
                "layer": layer_numbers[best_layer_index] if layer_numbers is not None else None,
                "layer_index": best_layer_index,
                "lambda": best_lambda,
                "inner_brier": best_score,
                "solver_converged": model.converged,
            }
        )
    if np.isnan(predictions).any():
        raise RuntimeError("Nested prediction did not fill every held-out row")
    return predictions, choices


def nested_baseline_predict(
    features: np.ndarray,
    y: np.ndarray,
    outer_folds: list[int],
    inner_fold_builder: Callable[[list[dict[str, Any]], int], list[int]],
    row_metadata: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = np.asarray(features, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    choices: list[dict[str, Any]] = []
    for outer_fold in sorted(set(outer_folds)):
        train_indices = [i for i, fold in enumerate(outer_folds) if fold != outer_fold]
        test_indices = [i for i, fold in enumerate(outer_folds) if fold == outer_fold]
        inner_folds = inner_fold_builder([row_metadata[i] for i in train_indices], outer_fold)
        lambda_value = select_lambda(features[train_indices], y[train_indices], inner_folds)
        model = fit_logistic(features[train_indices], y[train_indices], lambda_value)
        predictions[test_indices] = model.predict(features[test_indices])
        choices.append(
            {
                "outer_fold": outer_fold,
                "lambda": lambda_value,
                "solver_converged": model.converged,
            }
        )
    if np.isnan(predictions).any():
        raise RuntimeError("Baseline nested prediction did not fill every held-out row")
    return predictions, choices


def no_signal_predict(y: np.ndarray, outer_folds: list[int]) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    predictions = np.empty(len(y), dtype=np.float64)
    for fold in sorted(set(outer_folds)):
        train = np.asarray([i for i, value in enumerate(outer_folds) if value != fold])
        test = np.asarray([i for i, value in enumerate(outer_folds) if value == fold])
        predictions[test] = y[train].mean()
    return predictions


def select_final_hidden_hyperparameters(
    features: np.ndarray,
    y: np.ndarray,
    validation_folds: list[int],
    layer_numbers: list[int],
) -> dict[str, float | int]:
    features = np.asarray(features)
    y = np.asarray(y, dtype=np.int64)
    candidates: list[tuple[float, int, float]] = []
    for layer_index in range(features.shape[1]):
        layer_x = features[:, layer_index, :]
        for lambda_value in LAMBDA_GRID:
            validation_scores: list[float] = []
            for fold in sorted(set(validation_folds)):
                train = np.asarray([i for i, value in enumerate(validation_folds) if value != fold])
                validation = np.asarray(
                    [i for i, value in enumerate(validation_folds) if value == fold]
                )
                model = fit_logistic(layer_x[train], y[train], lambda_value)
                validation_scores.append(
                    brier_score(y[validation], model.predict(layer_x[validation]))
                )
            candidates.append((float(np.mean(validation_scores)), layer_index, lambda_value))
    best_brier, best_layer_index, best_lambda = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return {
        "layer_index": best_layer_index,
        "layer": int(layer_numbers[best_layer_index]),
        "lambda": best_lambda,
        "validation_brier": best_brier,
    }
