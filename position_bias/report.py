from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from position_bias.audit import check_budget
from position_bias.io_utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    judges = json.loads((args.analysis / "results.json").read_text())["judges"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prediction.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "judge",
                "n",
                "questions",
                "reversals",
                "excluded",
                "predictor",
                "auroc",
                "auroc_ci_low",
                "auroc_ci_high",
            ]
        )
        for label, result in judges.items():
            for name, entry in result["predictors"].items():
                ci = entry["bootstrap_95_ci"]["auroc"]
                writer.writerow(
                    [
                        label,
                        result["n"],
                        result.get("question_groups", result.get("questions")),
                        result["reversals"],
                        result["excluded"],
                        name,
                        entry["metrics"]["auroc"],
                        ci["lower"],
                        ci["upper"],
                    ]
                )
    n = len(judges)
    with (args.output_dir / "checking_budgets.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "judge",
                "predictor",
                "policy",
                "budget",
                "checked",
                "reversals",
                "detected",
                "recall",
                "precision",
                "calls",
                "call_savings_fraction",
            ]
        )
        for label, result in judges.items():
            for name, entry in result["predictors"].items():
                for policy in (
                    "source_budget_90",
                    "source_chosen_budget",
                    "fixed_50_percent",
                    "target_budget_90_retrospective",
                ):
                    values = entry.get(policy)
                    if values:
                        writer.writerow(
                            [
                                label,
                                name,
                                policy,
                                values["budget"],
                                values["checked"],
                                values["reversals"],
                                values["detected"],
                                values["recall"],
                                values["precision"],
                                values["judgment_calls"],
                                values["call_savings_fraction"],
                            ]
                        )
    write_json(
        args.output_dir / "coverage.json",
        {label: result.get("coverage") for label, result in judges.items()},
    )
    figure, axes = plt.subplots(1, n, figsize=(max(4, 3.5 * n), 3.5), squeeze=False, sharey=True)
    budgets = np.arange(101) / 100
    hyperparameters = {}
    for axis, (label, result) in zip(axes[0], judges.items()):
        rows = read_jsonl(args.analysis / label / "predictions.jsonl")
        y = np.asarray([r["y"] for r in rows])
        ids = [r["pair_id"] for r in rows]
        for name in ("probe", "output", "uncertainty", "output_uncertainty"):
            if name not in rows[0]["predictions"]:
                continue
            risk = np.asarray([r["predictions"][name] for r in rows])
            recall = [check_budget(y, risk, ids, b)["recall"] for b in budgets]
            if any(value is None for value in recall):
                continue
            axis.plot(budgets * 100, np.asarray(recall) * 100, label=name)
        axis.plot([0, 100], [0, 100], color="gray", linestyle="--", label="random (expected)")
        axis.set(
            title=label, xlabel="Pairs checked in reversed order (%)", xlim=(0, 100), ylim=(0, 100)
        )
        axis.grid(alpha=0.2)
        hyperparameters[label] = {
            name: {
                "outer_folds": entry.get("selected_outer_folds"),
                "final_refit": entry.get("selected_final_refit"),
            }
            for name, entry in result["predictors"].items()
        }
    axes[0, 0].set_ylabel("Reversals detected (%)")
    axes[0, -1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output_dir / "checking_curves.png", dpi=180)
    figure.savefig(args.output_dir / "checking_curves.svg")
    plt.close(figure)
    write_json(args.output_dir / "selected_hyperparameters.json", hyperparameters)

    write_json(
        args.output_dir / "provenance.json",
        {
            "analysis": str(args.analysis),
            "data": "computed predictions",
        },
    )


if __name__ == "__main__":
    main()
