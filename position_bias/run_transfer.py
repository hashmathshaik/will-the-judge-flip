from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from position_bias.artifacts import (
    aligned,
    check_frozen_protocol,
    coverage,
    load_judge,
    load_model,
    parse_spec,
    selected_rows,
    unique_specs,
)
from position_bias.audit import budget_interval, check_budget, recall_crossing
from position_bias.calibration import PlattScaler, ece
from position_bias.features import numeric_baselines
from position_bias.io_utils import write_json, write_jsonl
from position_bias.metrics import bootstrap_prediction_suite, score


def transfer(
    label: str,
    target_run: Path,
    source_analysis: Path,
    output: Path,
    bootstrap_reps: int = 2000,
    seed: int = 42,
    common_ids: set[str] | None = None,
) -> dict:
    judge = load_judge(target_run)
    frozen = json.loads((source_analysis / "frozen.json").read_text())
    check_frozen_protocol(frozen["protocol"], judge["manifest"])
    rows = selected_rows(judge, common_ids)
    hidden, y, metadata = aligned(judge, rows)
    matrices = numeric_baselines(metadata, rows)
    ids = [r["pair_id"] for r in rows]
    groups = np.asarray([r["group_id"] for r in rows])
    predictions, calibrated = {}, {}
    for name, entry in frozen["predictors"].items():
        model = load_model(source_analysis / entry["model_file"])
        if name == "probe":
            features = hidden[:, int(entry["selected"]["layer_index"]), :]
        elif name == "input_only":
            features = np.stack([judge["input_hidden_by_id"][p] for p in ids])[
                :, int(entry["selected"]["layer_index"]), :
            ]
        else:
            features = matrices[name]
        predictions[name] = model.predict(features)
        calibrated[name] = PlattScaler(**entry["platt_from_source_oof"]).predict(predictions[name])
    if not {"probe", "output", "output_uncertainty"}.issubset(predictions):
        raise ValueError("Frozen source release is missing a primary transfer predictor")

    comparator = max(
        ("output", "output_uncertainty"), key=lambda n: score(y, predictions[n])["auroc"] or -1
    )
    intervals = bootstrap_prediction_suite(
        y, predictions, groups, reference_name="probe", repetitions=bootstrap_reps, seed=seed
    )
    results = {
        "judge": label,
        "n": len(y),
        "questions": len(set(groups)),
        "reversals": int(y.sum()),
        "attempted": len(judge["records"]),
        "excluded": len(judge["records"]) - len(y),
        "target_fitting_or_calibration": False,
        "population": "common" if common_ids is not None else "per-judge",
        "best_output_comparator": comparator,
        "paired_gain_95_ci": intervals["reference_minus_predictor"][comparator],
        "predictors": {},
        "bootstrap": {
            "seed": seed,
            "repetitions": bootstrap_reps,
            "predictors_and_comparator_fixed": True,
        },
        "target_prevalence_constant_brier_retrospective": float(y.mean() * (1 - y.mean())),
    }
    for name, values in predictions.items():
        source_budget = frozen["predictors"][name]["source_budget_90"]
        results["predictors"][name] = {
            "metrics": score(y, values),
            "bootstrap_95_ci": intervals["predictors"][name],
            "source_calibrated_metrics": {
                **score(y, calibrated[name]),
                "ece": ece(y, calibrated[name]),
            },
            "fixed_50_percent": check_budget(y, values, ids, 0.5),
            "fixed_budgets": {str(b): check_budget(y, values, ids, b) for b in (0.25, 0.5, 0.75)},
            "source_chosen_budget": check_budget(y, values, ids, source_budget)
            if source_budget is not None
            else None,
            "target_budget_90_retrospective": recall_crossing(y, values, ids),
            "target_budget_90_95_ci": budget_interval(y, values, ids, groups, bootstrap_reps, seed),
        }
    slots = [r["ab"]["parse"]["slot"] for r in rows]
    results["slot_counts"] = {
        a + b: sum(r["ab"]["parse"]["slot"] == a and r["ba"]["parse"]["slot"] == b for r in rows)
        for a in ("A", "B")
        for b in ("A", "B")
    }
    results["confidence_subsets"] = {}
    results["first_slot_rates"] = {
        order: sum(r[order]["parse"]["slot"] == "A" for r in rows) / len(rows)
        for order in ("ab", "ba")
    }
    results["canonical_identity_labels_verified"] = len(rows)
    results["coverage"] = coverage(judge, rows)
    for threshold in (80, 90):
        indices = np.flatnonzero([m["confidence"] >= threshold for m in metadata])
        if len(indices):
            results["confidence_subsets"][str(threshold)] = check_budget(
                y[indices], predictions["probe"][indices], [ids[i] for i in indices], 0.5
            )
    from position_bias.metrics import score as metric_score

    results["probe_auroc_by_ab_slot"] = {
        slot: metric_score(
            y[np.asarray(slots) == slot], predictions["probe"][np.asarray(slots) == slot]
        )["auroc"]
        for slot in ("A", "B")
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "results.json", results)
    write_jsonl(
        output / "predictions.jsonl",
        [
            {
                "pair_id": p,
                "group_id": str(groups[i]),
                "y": int(y[i]),
                "predictions": {n: float(v[i]) for n, v in predictions.items()},
                "calibrated": {n: float(v[i]) for n, v in calibrated.items()},
            }
            for i, p in enumerate(ids)
        ],
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge", action="append", type=parse_spec, required=True, help="LABEL=TARGET_RUN"
    )
    parser.add_argument(
        "--source-analysis",
        type=Path,
        required=True,
        help="Source analysis root with one directory per judge",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--population", choices=("per-judge", "common"), default="per-judge")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    specs = unique_specs(args.judge)
    common = (
        set.intersection(*(load_judge(p, include_hidden=False)["usable"] for p in specs.values()))
        if args.population == "common"
        else None
    )
    results = {
        label: transfer(
            label,
            path,
            args.source_analysis / label,
            args.output_dir / label,
            args.bootstrap_reps,
            args.seed,
            common,
        )
        for label, path in specs.items()
    }
    write_json(args.output_dir / "results.json", {"judges": results, "target_fitting": False})


if __name__ == "__main__":
    main()
