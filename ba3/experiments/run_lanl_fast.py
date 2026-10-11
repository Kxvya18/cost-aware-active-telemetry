from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ba3.evaluation.metrics import ranking_metrics
from ba3.experiments.run_lanl import CandidateMatrix, MODALITIES, SourceRanker, make_matrix


def _copy_matrix(matrix):
    return CandidateMatrix(
        matrix.episode_id,
        matrix.candidates,
        matrix.truth,
        matrix.features.copy(),
        np.zeros_like(matrix.revealed),
        matrix.fisher_reliability,
        matrix.fisher_information,
    )


def _select(matrix, ranker, policy, fraction, costs, rng):
    """Batch acquisition uses one validation-trained utility ranking per decision round."""
    initial = ranker.probabilities(matrix)
    probabilities = np.asarray([initial[node] for node in matrix.candidates])
    activity = matrix.features[:, 0]
    structural = matrix.features[:, 4]
    structural_rank = np.argsort(np.argsort(structural)) / max(len(structural) - 1, 1)
    rows, modalities = np.indices((len(matrix.candidates), len(costs)))
    cost = costs[modalities]
    if policy == "random":
        utility = rng.random(rows.shape)
    elif policy == "activity":
        utility = np.repeat(activity[:, None], len(costs), axis=1)
    else:
        entropy_proxy = probabilities * (1.0 - probabilities)
        utility = np.repeat(entropy_proxy[:, None], len(costs), axis=1)
        if policy == "curvature_active":
            disagreement = np.abs(structural_rank - probabilities)
            utility *= (1.0 + matrix.fisher_reliability * disagreement)[:, None]
    ratio = (utility / cost).ravel()
    indices = np.argsort(-ratio, kind="stable")
    full_cost = len(matrix.candidates) * float(costs.sum())
    budget = max(float(costs.min()), full_cost * fraction / 100.0)
    spent = 0.0
    selected = 0
    for flat in indices:
        row, modality = divmod(int(flat), len(costs))
        if spent + costs[modality] > budget:
            continue
        matrix.revealed[row, [modality, 4 + modality]] = 1.0
        spent += costs[modality]
        selected += 1
    return spent, selected


def run(config_path: Path):
    config = json.loads(config_path.read_text())
    path = Path(config["processed_dir"]) / "episodes_900.pkl.gz"
    with gzip.open(path, "rb") as handle:
        records = pickle.load(handle)
    cap = int(config.get("evaluation_candidate_cap", 1000))
    matrices = [make_matrix(item, cap, config["fisher_scale"]) for item in records]
    matrices = [item for item in matrices if item is not None]
    train_boundary = int(0.6 * len(records))
    validation_boundary = int(0.8 * len(records))
    train = [item for item in matrices if item.episode_id < train_boundary]
    test = [item for item in matrices if item.episode_id >= validation_boundary]
    if not train or not test:
        raise RuntimeError("Candidate construction left an empty training or test partition")
    scale = float(np.median([item.fisher_information for item in train]))
    for item in matrices:
        raw = item.features[:, 4] / max(item.fisher_reliability, 1e-12)
        item.fisher_reliability = item.fisher_information / (item.fisher_information + scale)
        item.features[:, 4] = item.fisher_reliability * raw
    costs = np.asarray([config["modality_cost"][name] for name in MODALITIES], dtype=float)
    rows = []
    for seed in config["seeds"]:
        ranker = SourceRanker(seed).fit(train)
        for item in test:
            for policy in ("random", "activity", "entropy", "curvature_active"):
                for fraction in config["budgets_percent"]:
                    matrix = _copy_matrix(item)
                    spent, queries = _select(
                        matrix,
                        ranker,
                        policy,
                        fraction,
                        costs,
                        np.random.default_rng(seed * 100000 + matrix.episode_id),
                    )
                    scores = ranker.probabilities(matrix)
                    rows.append({
                        "seed": seed,
                        "episode_id": matrix.episode_id,
                        "policy": policy,
                        "budget_percent": fraction,
                        "cost": spent,
                        "queries": queries,
                        "fisher_reliability": matrix.fisher_reliability,
                        **ranking_metrics(scores, matrix.truth),
                    })
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "episode_results.csv", index=False)
    summary = frame.groupby(["policy", "budget_percent"]).agg(
        episodes=("episode_id", "nunique"), mrr=("mrr", "mean"), hit1=("hit1", "mean"),
        hit5=("hit5", "mean"), hit10=("hit10", "mean"), nll=("nll", "mean"),
        entropy=("entropy", "mean"), cost=("cost", "mean"), queries=("queries", "mean"),
    ).reset_index()
    summary.to_csv(output / "budget_summary.csv", index=False)
    audit = {
        "episodes_total": len(records), "episodes_candidate_recalled": len(matrices),
        "candidate_recall": len(matrices) / len(records), "train_episodes": len(train),
        "test_episodes": len(test), "evaluation_candidate_cap": cap,
        "curvature_scale_train_median": scale,
    }
    (output / "candidate_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config).to_string(index=False))


if __name__ == "__main__":
    main()
