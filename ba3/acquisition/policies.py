from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Query:
    host: str
    modality: str
    cost: float


def entropy(probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(probabilities * np.log(probabilities)))


def select_query(
    queries: list[Query],
    expected_posteriors: Callable[[Query], list[np.ndarray]],
    current_posterior: np.ndarray,
    policy: str,
    rng: np.random.Generator,
    activity: dict[str, float] | None = None,
    fisher_reliability: float = 0.0,
) -> Query:
    if not queries:
        raise ValueError("No admissible query remains")
    if policy == "random":
        return queries[int(rng.integers(0, len(queries)))]
    if policy == "uniform":
        return min(queries, key=lambda q: (q.cost, q.host, q.modality))
    if policy == "activity":
        activity = activity or {}
        return max(queries, key=lambda q: (activity.get(q.host, 0.0) / q.cost, -q.cost))

    current_entropy = entropy(current_posterior)
    best_query = queries[0]
    best_score = -np.inf
    for query in queries:
        outcomes = expected_posteriors(query)
        expected_entropy = float(np.mean([entropy(outcome) for outcome in outcomes]))
        information_gain = max(current_entropy - expected_entropy, 0.0)
        if policy == "entropy":
            score = information_gain / query.cost
        elif policy == "curvature_active":
            temporal_need = 1.0 - fisher_reliability
            disagreement_bonus = fisher_reliability * np.std([entropy(outcome) for outcome in outcomes])
            score = (temporal_need * information_gain + disagreement_bonus) / query.cost
        else:
            raise ValueError(f"Unknown policy {policy}")
        if score > best_score:
            best_score, best_query = score, query
    return best_query
