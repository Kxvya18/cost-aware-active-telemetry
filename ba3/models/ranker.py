from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "log_pre_auth",
    "log_in_degree",
    "log_out_degree",
    "log_pre_alert_contact",
    "boltzmann_log_prior",
    "log_post_auth",
    "log_post_dns",
    "log_post_flow",
    "log_post_proc",
    "log_post_alert_auth",
    "log_post_alert_dns",
    "log_post_alert_flow",
    "log_post_alert_proc",
)


@dataclass
class CandidateMatrix:
    episode_id: int
    candidates: list[str]
    truth: str
    features: np.ndarray
    revealed: np.ndarray
    fisher_reliability: float
    fisher_information: float = 0.0


class SourceRanker:
    def __init__(self, seed: int = 0):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )

    def fit(self, matrices: list[CandidateMatrix]) -> "SourceRanker":
        rng = np.random.default_rng(self.model.random_state)
        samples = []
        labels = []
        for item in matrices:
            label = np.asarray([candidate == item.truth for candidate in item.candidates], dtype=int)
            for fraction in (0.0, 0.01, 0.05, 0.25, 1.0):
                mask = rng.random(item.revealed.shape) < fraction
                augmented = CandidateMatrix(
                    item.episode_id,
                    item.candidates,
                    item.truth,
                    item.features,
                    mask.astype(float),
                    item.fisher_reliability,
                    item.fisher_information,
                )
                samples.append(self.mask_features(augmented))
                labels.append(label)
        x = np.vstack(samples)
        y = np.concatenate(labels)
        self.model.fit(self.scaler.fit_transform(x), y)
        return self

    @staticmethod
    def mask_features(matrix: CandidateMatrix) -> np.ndarray:
        values = matrix.features.copy()
        values[:, 5:] *= matrix.revealed
        return np.hstack([values, matrix.revealed])

    def probabilities(self, matrix: CandidateMatrix) -> dict[str, float]:
        logits = self.model.decision_function(self.scaler.transform(self.mask_features(matrix)))
        logits = logits - np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return dict(zip(matrix.candidates, probabilities.tolist()))

    def query_logit_delta(self, modality: int, value: float) -> float:
        """Logit change caused by revealing one previously hidden modality."""
        indices = [5 + modality, 9 + modality, 13 + modality, 17 + modality]
        raw_delta = np.asarray([value, value, 1.0, 1.0])
        scaled_delta = raw_delta / self.scaler.scale_[indices]
        return float(np.dot(self.model.coef_[0, indices], scaled_delta))
