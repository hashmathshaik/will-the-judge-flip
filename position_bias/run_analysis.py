from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from position_bias.folds import flip_burden_bin, make_grouped_folds
from position_bias.io_utils import read_jsonl, write_json, write_jsonl
from position_bias.metrics import (
    bootstrap_prediction_suite,
    bootstrap_stratified_suite,
    score,
    score_no_signal,
    stratified_auroc,
)
from position_bias.probe import nested_baseline_predict, nested_predict, no_signal_predict


def parse_judge_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Judge must be LABEL=RUN_DIRECTORY")
    label, directory = spec.split("=", maxsplit=1)
    if not label or not directory:
        raise argparse.ArgumentTypeError("Judge must be LABEL=RUN_DIRECTORY")
    return label, Path(directory)


def load_judge(run_directory: Path) -> dict[str, Any]:
    manifest = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    judgments = read_jsonl(run_directory / "judgments.jsonl")
    records = {row["pair_id"]: row for row in judgments}
    feature_metadata = read_jsonl(run_directory / "feature_metadata.jsonl")
    metadata = {row["pair_id"]: row for row in feature_metadata}
    feature_file = np.load(run_directory / "features.npz")
    feature_ids = [str(value) for value in feature_file["pair_ids"].tolist()]
    hidden = feature_file["hidden_states"]
    if len(feature_ids) != len(hidden) or len(set(feature_ids)) != len(feature_ids):
        raise ValueError(f"Feature IDs are missing or duplicated in {run_directory}")
    hidden_by_id = {pair_id: hidden[index] for index, pair_id in enumerate(feature_ids)}
    usable = {
        pair_id
        for pair_id, row in records.items()
        if row["ab"]["parse"]["format_ok"]
        and row["ba"]["parse"]["format_ok"]
        and pair_id in hidden_by_id
    }
    return {
        "manifest": manifest,
        "records": records,
        "metadata": metadata,
        "hidden_by_id": hidden_by_id,
        "layer_numbers": [int(value) for value in manifest["candidate_layers"]],
        "usable": usable,
        "run_directory": str(run_directory),
    }


def make_inner_folds(rows: list[dict[str, Any]], outer_fold: int, seed: int) -> list[int]:
    group_count = len({row["group_id"] for row in rows})
    return make_grouped_folds(
        rows,
        n_splits=min(3, group_count),
        seed=seed + 10_000 + outer_fold,
    )


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(f"{base}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def baseline_feature_matrices(
    metadata: list[dict[str, Any]], slots: list[str]
) -> dict[str, np.ndarray]:
    confidence = np.asarray([[float(item["confidence"])] for item in metadata])
    margin = np.asarray(
        [
            [
                float(item["signed_logit_margin"]),
                abs(float(item["signed_logit_margin"])),
            ]
            for item in metadata
        ]
    )
    length_difference = np.asarray(
        [
            [
                float(item["signed_length_difference"]),
                abs(float(item["signed_length_difference"])),
            ]
            for item in metadata
        ]
    )
    verdict_slot = np.asarray([[1.0 if slot == "X" else 0.0] for slot in slots])
    combined_output = np.hstack([confidence, margin, length_difference])
    return {
        "confidence": confidence,
        "logit_margin": margin,
        "length_difference": length_difference,
        "verdict_slot": verdict_slot,
        "combined_output": combined_output,
        "combined_output_slot": np.hstack([combined_output, verdict_slot]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grouped probes, baselines, metrics, and bootstrap intervals."
    )
    parser.add_argument(
        "--judge",
        action="append",
        required=True,
        type=parse_judge_spec,
        metavar="LABEL=RUN_DIRECTORY",
        help="Repeat once per completed judge run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.outer_folds < 2:
        raise ValueError("--outer-folds must be at least 2")
    if args.bootstrap_reps < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    judge_specs = dict(args.judge)
    if len(judge_specs) != len(args.judge):
        raise ValueError("Judge labels must be unique")
    judges = {label: load_judge(directory) for label, directory in judge_specs.items()}
    if len(judges) < 2:
        raise ValueError("The common intersection requires at least two judges")

    common_ids = set.intersection(*(judge["usable"] for judge in judges.values()))
    if not common_ids:
        raise ValueError("The judges have no common AB/BA parse-valid examples")
    first_label = next(iter(judges))
    first_records = judges[first_label]["records"]
    rows: list[dict[str, Any]] = []
    for pair_id in sorted(common_ids):
        base = first_records[pair_id]
        labels = {label: int(judge["records"][pair_id]["y_pos"]) for label, judge in judges.items()}
        row = {
            "pair_id": pair_id,
            "group_id": base["group_id"],
            "source": base.get("source", "unknown"),
            "coarse_domain": base.get("coarse_domain", "knowledge"),
            "labels": labels,
            "flip_burden": sum(labels.values()),
        }
        row["flip_burden_bin"] = flip_burden_bin(row["flip_burden"])
        rows.append(row)

    outer_folds = make_grouped_folds(rows, args.outer_folds, args.seed)
    for row, fold in zip(rows, outer_folds):
        row["outer_fold"] = fold

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "common_intersection.jsonl", rows)
    write_json(
        args.output_dir / "folds.json",
        {
            "seed": args.seed,
            "outer_folds": args.outer_folds,
            "pair_to_fold": {row["pair_id"]: row["outer_fold"] for row in rows},
            "group_to_fold": {row["group_id"]: row["outer_fold"] for row in rows},
            "strata": {
                "domain": sorted({row["coarse_domain"] for row in rows}),
                "flip_burden_bins": sorted({row["flip_burden_bin"] for row in rows}),
            },
        },
    )

    all_predictions: dict[str, dict[str, np.ndarray]] = {}
    results: dict[str, Any] = {
        "common_n": len(rows),
        "common_groups": len({row["group_id"] for row in rows}),
        "judges": {},
        "config": {
            "outer_folds": args.outer_folds,
            "inner_folds": 3,
            "bootstrap_reps": args.bootstrap_reps,
            "seed": args.seed,
        },
    }

    for label, judge in judges.items():
        y = np.asarray([row["labels"][label] for row in rows], dtype=np.int64)
        pfr = float(y.mean())
        slots = [str(judge["records"][row["pair_id"]]["ab"]["parse"]["slot"]) for row in rows]
        slot_array = np.asarray(slots)
        judge_result: dict[str, Any] = {
            "run_directory": judge["run_directory"],
            "n": len(y),
            "positive_count": int(y.sum()),
            "negative_count": int(len(y) - y.sum()),
            "pfr": pfr,
            "estimable": not (pfr < 0.05 or pfr > 0.95),
            "ab_slot_counts": {
                slot: int((slot_array == slot).sum()) for slot in sorted(set(slots))
            },
            "flip_rate_by_ab_slot": {
                slot: float(y[slot_array == slot].mean()) for slot in sorted(set(slots))
            },
            "predictors": {},
            "paired_probe_minus_baseline": {},
            "slot_stratified": {},
            "paired_probe_minus_baseline_slot_stratified": {},
        }
        if not judge_result["estimable"]:
            results["judges"][label] = judge_result
            continue

        hidden = np.stack([judge["hidden_by_id"][row["pair_id"]] for row in rows])
        metadata = [judge["metadata"][row["pair_id"]] for row in rows]
        outer_folds_list = [row["outer_fold"] for row in rows]

        def inner_builder(train_rows: list[dict[str, Any]], fold: int) -> list[int]:
            return make_inner_folds(train_rows, fold, args.seed)

        hidden_predictions, hidden_choices = nested_predict(
            hidden,
            y,
            outer_folds_list,
            inner_builder,
            rows,
            layer_numbers=judge["layer_numbers"],
        )
        baseline_inputs = baseline_feature_matrices(metadata, slots)
        predictions = {"hidden_state": hidden_predictions}
        choices: dict[str, Any] = {"hidden_state": hidden_choices}
        for predictor_name, predictor_features in baseline_inputs.items():
            predictions[predictor_name], choices[predictor_name] = nested_baseline_predict(
                predictor_features, y, outer_folds_list, inner_builder, rows
            )
        predictions["no_signal"] = no_signal_predict(y, outer_folds_list)
        choices["no_signal"] = [
            {
                "outer_fold": fold,
                "training_prevalence": float(
                    y[
                        np.asarray([i for i, value in enumerate(outer_folds_list) if value != fold])
                    ].mean()
                ),
            }
            for fold in sorted(set(outer_folds_list))
        ]
        all_predictions[label] = predictions

        groups = np.asarray([row["group_id"] for row in rows])
        bootstrap = bootstrap_prediction_suite(
            y,
            predictions,
            groups,
            reference_name="hidden_state",
            no_signal_names={"no_signal"},
            repetitions=args.bootstrap_reps,
            seed=stable_seed(label, args.seed),
        )
        for predictor_name, predictor_predictions in predictions.items():
            predictor_score = (
                score_no_signal(y, predictor_predictions)
                if predictor_name == "no_signal"
                else score(y, predictor_predictions)
            )
            predictor_result = {
                "metrics": predictor_score,
                "bootstrap_95_ci": bootstrap["predictors"][predictor_name],
            }
            predictor_result["selected_outer_folds"] = choices[predictor_name]
            judge_result["predictors"][predictor_name] = predictor_result

        for baseline_name in (
            "confidence",
            "logit_margin",
            "length_difference",
            "verdict_slot",
            "combined_output",
            "combined_output_slot",
            "no_signal",
        ):
            judge_result["paired_probe_minus_baseline"][baseline_name] = bootstrap[
                "reference_minus_predictor"
            ][baseline_name]

        stratified_predictions = {
            name: values for name, values in predictions.items() if name != "no_signal"
        }
        stratified = bootstrap_stratified_suite(
            y,
            stratified_predictions,
            groups,
            slot_array,
            reference_name="hidden_state",
            repetitions=args.bootstrap_reps,
            seed=stable_seed(label, args.seed),
        )
        for predictor_name, predictor_predictions in stratified_predictions.items():
            judge_result["slot_stratified"][predictor_name] = {
                "auroc": stratified_auroc(y, predictor_predictions, slot_array),
                "bootstrap_95_ci": stratified["predictors"][predictor_name],
            }
        for baseline_name, interval in stratified["reference_minus_predictor"].items():
            judge_result["paired_probe_minus_baseline_slot_stratified"][baseline_name] = interval
        results["judges"][label] = judge_result

    prediction_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prediction_row: dict[str, Any] = {
            "pair_id": row["pair_id"],
            "group_id": row["group_id"],
            "outer_fold": row["outer_fold"],
        }
        for label, judge_predictions in all_predictions.items():
            prediction_row[label] = {
                "y_pos": row["labels"][label],
                **{
                    predictor: float(values[index])
                    for predictor, values in judge_predictions.items()
                },
            }
        prediction_rows.append(prediction_row)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_json(args.output_dir / "results.json", results)

    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "judge",
                "predictor",
                "n",
                "pfr",
                "auroc",
                "auprc",
                "brier",
                "slot_stratified_auroc",
            ]
        )
        for label, judge_result in results["judges"].items():
            if not judge_result["estimable"]:
                writer.writerow(
                    [
                        label,
                        "NE",
                        judge_result["n"],
                        judge_result["pfr"],
                        "NE",
                        "NE",
                        "NE",
                        "NE",
                    ]
                )
                continue
            for predictor, predictor_result in judge_result["predictors"].items():
                metrics = predictor_result["metrics"]
                stratified_entry = judge_result["slot_stratified"].get(predictor)
                writer.writerow(
                    [
                        label,
                        predictor,
                        judge_result["n"],
                        judge_result["pfr"],
                        metrics["auroc"],
                        metrics["auprc"],
                        metrics["brier"],
                        stratified_entry["auroc"] if stratified_entry else "",
                    ]
                )

    print(f"Common intersection: {len(rows)} pairs, {results['common_groups']} groups")
    for label, judge_result in results["judges"].items():
        print(f"\n{label}: PFR={judge_result['pfr']:.3f} N={judge_result['n']}")
        for predictor, predictor_result in judge_result.get("predictors", {}).items():
            metrics = predictor_result["metrics"]
            stratified_entry = judge_result["slot_stratified"].get(predictor)
            stratified_value = (
                f" slotAUROC={stratified_entry['auroc']:.3f}"
                if stratified_entry and stratified_entry["auroc"] is not None
                else ""
            )
            print(
                f"  {predictor:22s} AUROC={metrics['auroc']} "
                f"AUPRC={metrics['auprc']} Brier={metrics['brier']}{stratified_value}"
            )


if __name__ == "__main__":
    main()
