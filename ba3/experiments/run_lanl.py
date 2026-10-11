from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ba3.data.preprocess import EpisodeTelemetry
from ba3.evaluation.metrics import ranking_metrics
from ba3.models.ranker import CandidateMatrix, SourceRanker


MODALITIES = ("auth", "dns", "flow", "proc")


def _log1p(counter, key):
    return np.log1p(float(counter.get(key, 0.0)))


def make_matrix(record: EpisodeTelemetry, candidate_cap: int, fisher_scale: float) -> CandidateMatrix | None:
    activity = record.pre_activity.get("auth", {})
    candidates = [node for node, _ in activity.most_common(candidate_cap) if node.startswith("C")]
    if record.episode.hidden_origin not in candidates:
        return None
    total_degree = record.pre_in_degree + record.pre_out_degree
    degree_values = np.asarray([total_degree.get(node, 0.0) for node in candidates], dtype=float)
    log_degree = np.log1p(degree_values)
    fisher = float(len(candidates) * np.var(log_degree))
    reliability = fisher / (fisher + fisher_scale)
    beta = 1.0
    features = []
    for node in candidates:
        row = [
            _log1p(record.pre_activity["auth"], node),
            _log1p(record.pre_in_degree, node),
            _log1p(record.pre_out_degree, node),
            _log1p(record.pre_alert_contacts, node),
            reliability * beta * np.log(float(total_degree.get(node, 0.0)) + 1.0),
        ]
        row.extend(_log1p(record.post_activity[modality], node) for modality in MODALITIES)
        row.extend(_log1p(record.post_alert_contacts[modality], node) for modality in MODALITIES)
        features.append(row)
    return CandidateMatrix(
        episode_id=record.episode.episode_id,
        candidates=candidates,
        truth=record.episode.hidden_origin,
        features=np.asarray(features, dtype=float),
        revealed=np.zeros((len(candidates), 8), dtype=float),
        fisher_reliability=reliability,
        fisher_information=fisher,
    )


def posterior_entropy(probabilities: dict[str, float]) -> float:
    values = np.asarray(list(probabilities.values()), dtype=float)
    return float(-np.sum(values * np.log(np.clip(values, 1e-12, 1.0))))


def entropy_after_logit_delta(probability, entropy_before, delta):
    factor = np.exp(np.clip(delta, -30.0, 30.0))
    normalizer = 1.0 - probability + probability * factor
    selected = probability * factor / normalizer
    other_mass = (1.0 - probability) / normalizer
    selected_term = -selected * np.log(max(selected, 1e-12))
    old_selected_term = -probability * np.log(max(probability, 1e-12))
    old_other_entropy = entropy_before - old_selected_term
    other_entropy = old_other_entropy / normalizer + other_mass * np.log(normalizer)
    return float(selected_term + other_entropy)


def simulated_gain(ranker, probability, entropy_before, modality, nonzero_probability, nonzero_value):
    zero_entropy = entropy_after_logit_delta(
        probability, entropy_before, ranker.query_logit_delta(modality, 0.0)
    )
    nonzero_entropy = entropy_after_logit_delta(
        probability, entropy_before, ranker.query_logit_delta(modality, nonzero_value)
    )
    expected = (1.0 - nonzero_probability) * zero_entropy + nonzero_probability * nonzero_entropy
    return max(entropy_before - expected, 0.0)


def acquire(
    matrix,
    ranker,
    cost_budget,
    policy,
    rng,
    costs,
    outcome_model,
):
    spent = 0.0
    queries = 0
    queried = set()
    candidate_index = {node: index for index, node in enumerate(matrix.candidates)}
    while True:
        admissible = [
            (row, modality)
            for row in range(len(matrix.candidates))
            for modality in range(4)
            if (row, modality) not in queried and spent + costs[modality] <= cost_budget
        ]
        if not admissible:
            break
        posterior = ranker.probabilities(matrix)
        if policy == "random":
            choice = admissible[int(rng.integers(0, len(admissible)))]
        else:
            scores = []
            entropy_before = posterior_entropy(posterior)
            structural = matrix.features[:, 4]
            structural_rank = np.argsort(np.argsort(structural)) / max(len(structural) - 1, 1)
            for row, modality in admissible:
                node = matrix.candidates[row]
                if policy == "activity":
                    score = matrix.features[row, 0] / costs[modality]
                else:
                    prevalence, nonzero_value = outcome_model[modality]
                    gain = simulated_gain(
                        ranker,
                        posterior[node],
                        entropy_before,
                        modality,
                        prevalence,
                        nonzero_value,
                    )
                    if policy == "entropy":
                        score = gain / costs[modality]
                    elif policy == "curvature_active":
                        disagreement = abs(structural_rank[row] - posterior[node])
                        score = gain * (1.0 + matrix.fisher_reliability * disagreement) / costs[modality]
                    else:
                        raise ValueError(policy)
                scores.append(score)
            choice = admissible[int(np.argmax(scores))]
        row, modality = choice
        matrix.revealed[row, [modality, 4 + modality]] = 1.0
        queried.add(choice)
        spent += costs[modality]
        queries += 1
    return spent, queries


def run(config_path: Path) -> pd.DataFrame:
    config = json.loads(config_path.read_text())
    processed = Path(config["processed_dir"]) / "episodes_900.pkl.gz"
    with gzip.open(processed, "rb") as handle:
        records: list[EpisodeTelemetry] = pickle.load(handle)
    attempted = [make_matrix(record, config["candidate_cap"], config["fisher_scale"]) for record in records]
    matrices = [matrix for matrix in attempted if matrix is not None]
    matrices.sort(key=lambda item: item.episode_id)
    train_boundary = int(0.6 * len(records))
    validation_boundary = int(0.8 * len(records))
    train = [item for item in matrices if item.episode_id < train_boundary]
    validation = [item for item in matrices if train_boundary <= item.episode_id < validation_boundary]
    test = [item for item in matrices if item.episode_id >= validation_boundary]
    curvature_scale = float(np.median([item.fisher_information for item in train]))
    for item in matrices:
        old_reliability = max(item.fisher_reliability, 1e-12)
        raw_prior = item.features[:, 4] / old_reliability
        item.fisher_reliability = item.fisher_information / (item.fisher_information + curvature_scale)
        item.features[:, 4] = item.fisher_reliability * raw_prior
    costs = np.asarray([config["modality_cost"][name] for name in MODALITIES], dtype=float)
    outcome_model = []
    for modality in range(4):
        values = np.concatenate([item.features[:, [5 + modality, 9 + modality]].ravel() for item in train])
        prevalence = float(np.mean(values > 0))
        nonzero = values[values > 0]
        outcome_model.append((prevalence, float(np.median(nonzero)) if len(nonzero) else 1.0))
    rows = []
    for seed in config["seeds"]:
        ranker = SourceRanker(seed).fit(train)
        for original in test:
            fixed_probabilities = {}
            degree = np.exp(original.features[:, 1] + original.features[:, 2])
            fixed_probabilities["degree"] = degree / degree.sum()
            contact = np.exp(original.features[:, 3])
            fixed_probabilities["alert_contact"] = contact / contact.sum()
            blind = np.exp(original.features[:, 4] / max(original.fisher_reliability, 1e-6))
            fixed_probabilities["boltzmann_only"] = blind / blind.sum()
            fixed_probabilities["static_audited"] = np.asarray(
                list(ranker.probabilities(original).values()), dtype=float
            )
            for method, values in fixed_probabilities.items():
                posterior = dict(zip(original.candidates, values.tolist()))
                metrics = ranking_metrics(posterior, original.truth)
                rows.append(
                    {
                        "seed": seed,
                        "episode_id": original.episode_id,
                        "policy": method,
                        "budget_percent": 0.0,
                        "queries": 0,
                        "cost": 0.0,
                        "fisher_reliability": original.fisher_reliability,
                        **metrics,
                    }
                )
            for policy in ("random", "activity", "entropy", "curvature_active"):
                for percent in config["budgets_percent"]:
                    matrix = CandidateMatrix(
                        original.episode_id,
                        original.candidates,
                        original.truth,
                        original.features,
                        np.zeros_like(original.revealed),
                        original.fisher_reliability,
                        original.fisher_information,
                    )
                    posterior = ranker.probabilities(matrix)
                    full_cost = len(matrix.candidates) * float(costs.sum())
                    cost_budget = max(float(costs.min()), full_cost * percent / 100.0)
                    spent, queries = acquire(
                        matrix,
                        ranker,
                        cost_budget,
                        policy,
                        np.random.default_rng(seed + matrix.episode_id),
                        costs,
                        outcome_model,
                    )
                    posterior = ranker.probabilities(matrix)
                    metrics = ranking_metrics(posterior, matrix.truth)
                    rows.append(
                        {
                            "seed": seed,
                            "episode_id": matrix.episode_id,
                            "policy": policy,
                            "budget_percent": percent,
                            "queries": queries,
                            "cost": spent,
                            "fisher_reliability": matrix.fisher_reliability,
                            **metrics,
                        }
                    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "episodes_total": len(records),
        "episodes_candidate_recalled": len(matrices),
        "candidate_recall": len(matrices) / max(len(records), 1),
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "test_episodes": len(test),
        "candidate_cap": config["candidate_cap"],
        "curvature_scale_train_median": curvature_scale,
    }
    (output_dir / "candidate_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "episode_results.csv", index=False)
    summary = frame.groupby(["policy", "budget_percent"]).agg(
        episodes=("episode_id", "nunique"),
        mrr=("mrr", "mean"),
        hit1=("hit1", "mean"),
        hit5=("hit5", "mean"),
        hit10=("hit10", "mean"),
        normalized_rank=("normalized_rank", "median"),
        nll=("nll", "mean"),
        entropy=("entropy", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "budget_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config).to_string(index=False))


if __name__ == "__main__":
    main()
