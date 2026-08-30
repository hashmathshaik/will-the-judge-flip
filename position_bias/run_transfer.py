from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from position_bias.folds import flip_burden_bin, make_grouped_folds
from position_bias.io_utils import write_json, write_jsonl
from position_bias.metrics import (
    bootstrap_prediction_suite,
    score,
    score_no_signal,
)
from position_bias.probe import fit_logistic, select_final_hidden_hyperparameters
from position_bias.run_analysis import load_judge, parse_judge_spec


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(f"{base}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def frozen_no_signal_predictions(y_id: np.ndarray, ood_size: int) -> np.ndarray:
    return np.full(ood_size, float(np.asarray(y_id).mean()), dtype=np.float64)


def common_rows(judges: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    common_ids = set.intersection(*(judge["usable"] for judge in judges.values()))
    if not common_ids:
        raise ValueError("No common parse-valid AB/BA examples")
    first = next(iter(judges.values()))
    rows: list[dict[str, Any]] = []
    for pair_id in sorted(common_ids):
        base = first["records"][pair_id]
        labels = {label: int(judge["records"][pair_id]["y_pos"]) for label, judge in judges.items()}
        burden = sum(labels.values())
        rows.append(
            {
                "pair_id": pair_id,
                "group_id": base["group_id"],
                "source": base.get("source", "unknown"),
                "coarse_domain": base.get("coarse_domain", "unknown"),
                "labels": labels,
                "flip_burden": burden,
                "flip_burden_bin": flip_burden_bin(burden),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit linear probes on JudgeBench and evaluate them on MT-Bench."
    )
    parser.add_argument(
        "--id-judge",
        action="append",
        required=True,
        type=parse_judge_spec,
        metavar="LABEL=JUDGEBENCH_RUN_DIRECTORY",
    )
    parser.add_argument(
        "--ood-judge",
        action="append",
        required=True,
        type=parse_judge_spec,
        metavar="LABEL=MTBENCH_RUN_DIRECTORY",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-folds", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    id_specs = dict(args.id_judge)
    ood_specs = dict(args.ood_judge)
    if set(id_specs) != set(ood_specs):
        raise ValueError("ID and OOD judge labels must match exactly")
    if len(id_specs) != len(args.id_judge) or len(ood_specs) != len(args.ood_judge):
        raise ValueError("Judge labels must be unique")

    id_judges = {label: load_judge(path) for label, path in id_specs.items()}
    ood_judges = {label: load_judge(path) for label, path in ood_specs.items()}
    id_rows = common_rows(id_judges)
    ood_rows = common_rows(ood_judges)
    selection_folds = make_grouped_folds(id_rows, n_splits=args.selection_folds, seed=args.seed)

    results: dict[str, Any] = {
        "id_benchmark": "JudgeBench",
        "ood_benchmark": "MT-Bench",
        "id_common_n": len(id_rows),
        "ood_common_n": len(ood_rows),
        "ood_question_groups": len({row["group_id"] for row in ood_rows}),
        "selection_uses": "JudgeBench only",
        "mtbench_fitting_or_recalibration": False,
        "judges": {},
        "config": {
            "selection_folds": args.selection_folds,
            "bootstrap_reps": args.bootstrap_reps,
            "seed": args.seed,
        },
    }
    prediction_rows = [
        {
            "pair_id": row["pair_id"],
            "group_id": row["group_id"],
        }
        for row in ood_rows
    ]

    for label in id_specs:
        y_id = np.asarray([row["labels"][label] for row in id_rows], dtype=np.int64)
        y_ood = np.asarray([row["labels"][label] for row in ood_rows], dtype=np.int64)
        id_pfr = float(y_id.mean())
        ood_pfr = float(y_ood.mean())
        id_estimable = not (id_pfr < 0.05 or id_pfr > 0.95)
        ood_estimable = not (ood_pfr < 0.05 or ood_pfr > 0.95)
        judge_result: dict[str, Any] = {
            "id_n": len(y_id),
            "id_positive_count": int(y_id.sum()),
            "id_pfr": id_pfr,
            "id_estimable": id_estimable,
            "ood_n": len(y_ood),
            "ood_positive_count": int(y_ood.sum()),
            "ood_negative_count": int(len(y_ood) - y_ood.sum()),
            "ood_pfr": ood_pfr,
            "ood_estimable": ood_estimable,
        }
        if not id_estimable:
            judge_result["status"] = "JudgeBench target degenerate; transfer not fit"
            results["judges"][label] = judge_result
            continue

        id_hidden = np.stack([id_judges[label]["hidden_by_id"][row["pair_id"]] for row in id_rows])
        ood_hidden = np.stack(
            [ood_judges[label]["hidden_by_id"][row["pair_id"]] for row in ood_rows]
        )
        id_layers = id_judges[label]["layer_numbers"]
        ood_layers = ood_judges[label]["layer_numbers"]
        if id_layers != ood_layers:
            raise ValueError(f"Candidate layers differ between ID and OOD for {label}")

        selected = select_final_hidden_hyperparameters(id_hidden, y_id, selection_folds, id_layers)
        layer_index = int(selected["layer_index"])
        model = fit_logistic(id_hidden[:, layer_index, :], y_id, float(selected["lambda"]))
        if not model.converged:
            raise RuntimeError(f"Final JudgeBench fit did not converge for {label}")
        hidden_predictions = model.predict(ood_hidden[:, layer_index, :])
        no_signal_predictions = frozen_no_signal_predictions(y_id, len(y_ood))
        judge_result["selected_on_judgebench"] = selected
        judge_result["frozen_no_signal_probability"] = id_pfr

        if ood_estimable:
            groups = np.asarray([row["group_id"] for row in ood_rows])
            bootstrap = bootstrap_prediction_suite(
                y_ood,
                {
                    "hidden_state": hidden_predictions,
                    "no_signal": no_signal_predictions,
                },
                groups,
                reference_name="hidden_state",
                no_signal_names={"no_signal"},
                repetitions=args.bootstrap_reps,
                seed=stable_seed(label, args.seed),
            )
            judge_result["hidden_state"] = {
                "metrics": score(y_ood, hidden_predictions),
                "bootstrap_95_ci": bootstrap["predictors"]["hidden_state"],
            }
            judge_result["no_signal"] = {
                "metrics": score_no_signal(y_ood, no_signal_predictions),
                "bootstrap_95_ci": bootstrap["predictors"]["no_signal"],
            }
            judge_result["paired_hidden_minus_no_signal"] = bootstrap["reference_minus_predictor"][
                "no_signal"
            ]
        else:
            judge_result["hidden_state"] = {
                "metrics": {"pfr": ood_pfr, "auroc": None, "auprc": None, "brier": None}
            }
            no_signal_brier = float(np.mean((no_signal_predictions - y_ood) ** 2))
            judge_result["no_signal"] = {
                "metrics": {
                    "pfr": ood_pfr,
                    "auroc": None,
                    "auprc": None,
                    "brier": no_signal_brier,
                }
            }

        for index, row in enumerate(prediction_rows):
            row[label] = {
                "y_pos": int(y_ood[index]),
                "hidden_state": float(hidden_predictions[index]),
                "no_signal": float(no_signal_predictions[index]),
            }
        results["judges"][label] = judge_result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "mtbench_common_intersection.jsonl", ood_rows)
    write_jsonl(args.output_dir / "mtbench_predictions.jsonl", prediction_rows)
    write_json(args.output_dir / "transfer_results.json", results)
    print(
        f"JudgeBench common N={len(id_rows)}; MT-Bench common N={len(ood_rows)} "
        f"over {results['ood_question_groups']} question groups."
    )
    for label, judge_result in results["judges"].items():
        metrics = judge_result.get("hidden_state", {}).get("metrics", {})
        print(
            f"{label}: ID PFR={judge_result['id_pfr']:.3f}; "
            f"OOD PFR={judge_result['ood_pfr']:.3f}; "
            f"OOD AUROC={metrics.get('auroc')} Brier={metrics.get('brier')}"
        )


if __name__ == "__main__":
    main()
