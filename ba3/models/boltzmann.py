from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def sequential_log_likelihood(beta: float, choices: list[int], degree_snapshots: list[np.ndarray], alpha: float = 1.0) -> float:
    """Sequential without-replacement likelihood over observed attachment choices."""
    total = 0.0
    available = None
    for choice, degrees in zip(choices, degree_snapshots):
        degrees = np.asarray(degrees, dtype=float)
        if available is None or len(available) != len(degrees):
            available = np.ones(len(degrees), dtype=bool)
        logits = beta * np.log(degrees + alpha)
        logits[~available] = -np.inf
        finite = np.isfinite(logits)
        shift = np.max(logits[finite])
        total += logits[choice] - shift - np.log(np.exp(logits[finite] - shift).sum())
        available[choice] = False
    return float(total)


def fit_beta(choices: list[int], degree_snapshots: list[np.ndarray], alpha: float = 1.0) -> float:
    result = minimize_scalar(
        lambda beta: -sequential_log_likelihood(beta, choices, degree_snapshots, alpha),
        bounds=(-2.0, 4.0),
        method="bounded",
    )
    return float(result.x)


def exact_observed_fisher(beta: float, choices: list[int], degree_snapshots: list[np.ndarray], alpha: float = 1.0) -> float:
    """Observed curvature for the sequential without-replacement likelihood."""
    information = 0.0
    available = None
    for choice, degrees in zip(choices, degree_snapshots):
        degrees = np.asarray(degrees, dtype=float)
        if available is None or len(available) != len(degrees):
            available = np.ones(len(degrees), dtype=bool)
        log_degree = np.log(degrees + alpha)
        logits = beta * log_degree
        logits[~available] = -np.inf
        finite = np.isfinite(logits)
        weights = np.exp(logits[finite] - np.max(logits[finite]))
        probabilities = weights / weights.sum()
        values = log_degree[finite]
        information += float(np.dot(probabilities, values**2) - np.dot(probabilities, values) ** 2)
        available[choice] = False
    return max(information, 0.0)


def fisher_reliability(information: float, scale: float = 25.0) -> float:
    return float(information / (information + scale)) if information > 0 else 0.0
