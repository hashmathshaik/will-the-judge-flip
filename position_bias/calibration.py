from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


@dataclass
class PlattScaler:
    slope: float
    intercept: float

    @classmethod
    def fit(cls, probabilities: np.ndarray, labels: np.ndarray) -> "PlattScaler":
        x = logit(np.clip(probabilities, 1e-6, 1 - 1e-6))
        y = np.asarray(labels)
        if len(x) != len(y) or not len(y) or not np.isin(y, [0, 1]).all():
            raise ValueError("Calibration requires aligned nonempty binary source labels")
        if len(np.unique(y)) < 2:
            return cls(0.0, float(logit(np.clip(y.mean(), 1e-6, 1 - 1e-6))))

        def objective(p):
            z = p[0] * x + p[1]
            return float(np.mean(np.logaddexp(0, z) - y * z) + 1e-8 * (p @ p))

        fit = minimize(objective, np.array([1.0, 0.0]), method="L-BFGS-B")
        if not fit.success:
            raise RuntimeError(f"Platt scaling failed: {fit.message}")
        return cls(float(fit.x[0]), float(fit.x[1]))

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return expit(self.slope * logit(np.clip(probabilities, 1e-6, 1 - 1e-6)) + self.intercept)


def ece(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(probabilities)
    if not len(y) or len(y) != len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("ECE requires aligned nonempty labels and probabilities in [0,1]")
    assignments = np.minimum((p * bins).astype(int), bins - 1)
    return float(
        sum(
            np.sum(assignments == i)
            / len(y)
            * abs(p[assignments == i].mean() - y[assignments == i].mean())
            for i in range(bins)
            if np.any(assignments == i)
        )
    )
