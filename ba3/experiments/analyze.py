from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ba3.evaluation.metrics import paired_bootstrap, paired_permutation_test


def analyze(result_csv: Path, output_dir: Path, repetitions: int = 2000) -> dict:
    frame = pd.read_csv(result_csv)
    active = frame[frame.policy.isin(["curvature_active", "entropy"])]
    comparisons = []
    for budget in sorted(active.budget_percent.unique()):
        subset = active[active.budget_percent == budget]
        pivot = subset.pivot_table(index=["seed", "episode_id"], columns="policy", values="mrr")
        pivot = pivot.dropna()
        delta, low, high = paired_bootstrap(
            pivot["curvature_active"].to_numpy(),
            pivot["entropy"].to_numpy(),
            repetitions=repetitions,
            seed=int(1000 * budget),
        )
        p_value = paired_permutation_test(
            pivot["curvature_active"].to_numpy(), pivot["entropy"].to_numpy(), seed=int(1000 * budget)
        )
        comparisons.append(
            {
                "budget_percent": float(budget),
                "pairs": int(len(pivot)),
                "mrr_delta": delta,
                "ci_low": low,
                "ci_high": high,
                "permutation_p": p_value,
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output_dir / "paired_policy_comparison.csv", index=False)

    curves = active.pivot_table(
        index=["seed", "episode_id", "policy"], columns="budget_percent", values="mrr"
    ).reset_index()
    budgets = np.asarray(sorted(active.budget_percent.unique()), dtype=float) / 100.0
    aubc_rows = []
    for _, row in curves.iterrows():
        values = np.asarray([row[100.0 * value] for value in budgets], dtype=float)
        aubc_rows.append(
            {
                "seed": int(row.seed),
                "episode_id": int(row.episode_id),
                "policy": row.policy,
                "aubc": float(np.trapezoid(values, budgets) / (budgets[-1] - budgets[0])),
            }
        )
    aubc = pd.DataFrame(aubc_rows)
    aubc.to_csv(output_dir / "aubc.csv", index=False)
    aubc_pivot = aubc.pivot(index=["seed", "episode_id"], columns="policy", values="aubc").dropna()
    aubc_delta, aubc_low, aubc_high = paired_bootstrap(
        aubc_pivot["curvature_active"], aubc_pivot["entropy"], repetitions=repetitions, seed=991
    )
    decision = {
        "comparison": "curvature_active_minus_entropy",
        "aubc_delta": aubc_delta,
        "aubc_ci_low": aubc_low,
        "aubc_ci_high": aubc_high,
        "aubc_permutation_p": paired_permutation_test(
            aubc_pivot["curvature_active"], aubc_pivot["entropy"], seed=991
        ),
        "central_claim_supported": bool(aubc_low > 0.0),
        "interpretation": (
            "The pre-specified active-telemetry claim passed the held-out paired test."
            if aubc_low > 0.0
            else "The pre-specified active-telemetry claim did not pass the held-out paired test."
        ),
    }
    (output_dir / "claim_audit.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.results, args.output_dir, args.repetitions), indent=2))


if __name__ == "__main__":
    main()
