from __future__ import annotations

import numpy as np


def source_rank(probabilities: dict[str, float], truth: str) -> int:
    ordered = sorted(probabilities, key=lambda node: (-probabilities[node], node))
    return ordered.index(truth) + 1 if truth in probabilities else len(ordered) + 1


def ranking_metrics(probabilities: dict[str, float], truth: str) -> dict[str, float]:
    rank = source_rank(probabilities, truth)
    n = max(len(probabilities), 1)
    p = float(probabilities.get(truth, 0.0))
    return {
        "rank": float(rank),
        "mrr": 1.0 / rank,
        "hit1": float(rank <= 1),
        "hit5": float(rank <= 5),
        "hit10": float(rank <= 10),
        "normalized_rank": (rank - 1.0) / max(n - 1.0, 1.0),
        "nll": -np.log(max(p, 1e-12)),
        "brier": sum((value - float(node == truth)) ** 2 for node, value in probabilities.items()),
        "entropy": -sum(value * np.log(max(value, 1e-12)) for value in probabilities.values()),
    }


def paired_bootstrap(values_a, values_b, repetitions: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Paired samples must have equal length")
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions)
    for index in range(repetitions):
        sample = rng.integers(0, len(a), len(a))
        deltas[index] = np.mean(a[sample] - b[sample])
    return float(np.mean(a - b)), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def paired_permutation_test(values_a, values_b, repetitions: int = 10000, seed: int = 0) -> float:
    delta = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    observed = abs(float(delta.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(repetitions, len(delta)))
    null = np.abs((signs * delta).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (repetitions + 1))
