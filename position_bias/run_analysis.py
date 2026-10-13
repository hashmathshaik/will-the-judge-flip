from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from position_bias.artifacts import (
    aligned,
    coverage,
    load_judge,
    parse_spec,
    save_model,
    selected_rows,
    unique_specs,
)
from position_bias.audit import budget_interval, check_budget, recall_crossing
from position_bias.calibration import PlattScaler, ece
from position_bias.features import TFIDF_SETTINGS, all_baselines, numeric_baselines
from position_bias.folds import make_grouped_folds
from position_bias.io_utils import write_json, write_jsonl
from position_bias.metrics import bootstrap_prediction_suite, score
from position_bias.probe import (
    fit_logistic,
    nested_baseline_predict,
    nested_predict,
    select_final_hidden_hyperparameters,
    select_lambda,
)


def analyze(
    label: str,
    directory: Path,
    output: Path,
    embeddings: Path | None,
    numeric_only: bool = False,
    common_ids: set[str] | None = None,
    bootstrap_reps: int = 2000,
    seed: int = 42,
) -> dict:
    started = time.perf_counter()
    judge = load_judge(directory)
    rows = selected_rows(judge, common_ids)
    hidden, y, metadata = aligned(judge, rows)
    ids = [r["pair_id"] for r in rows]
    groups = np.asarray([r["group_id"] for r in rows])
    outer = make_grouped_folds(rows, 5, seed)

    def inner(train_rows, fold):
        return make_grouped_folds(train_rows, 3, seed + fold + 1000)

    predictions, choices = {}, {}
    predictions["probe"], choices["probe"] = nested_predict(
        hidden, y, outer, inner, rows, layer_numbers=judge["layer_numbers"]
    )
    matrices = (
        numeric_baselines(metadata, rows)
        if numeric_only
        else all_baselines(metadata, rows, embeddings)
    )
    for name, features in matrices.items():
        print(f"{label}: fitting {name}", flush=True)
        predictions[name], choices[name] = nested_baseline_predict(features, y, outer, inner, rows)
    if judge["input_hidden_by_id"]:
        input_hidden = np.stack([judge["input_hidden_by_id"][p] for p in ids])
        predictions["input_only"], choices["input_only"] = nested_predict(
            input_hidden, y, outer, inner, rows, layer_numbers=judge["layer_numbers"]
        )
    intervals = bootstrap_prediction_suite(
        y, predictions, groups, reference_name="probe", repetitions=bootstrap_reps, seed=seed
    )
    results = {
        "judge": label,
        "n": len(rows),
        "question_groups": len(set(groups)),
        "attempted": len(judge["records"]),
        "excluded": len(judge["records"]) - len(rows),
        "reversals": int(y.sum()),
        "reversal_rate": float(y.mean()),
        "population": "common-intersection" if common_ids is not None else "per-judge-retained",
        "full_ten_baselines": not numeric_only,
        "protocol": judge["manifest"],
        "coverage": coverage(judge, rows),
        "configuration": {
            "outer_folds": 5,
            "inner_folds": 3,
            "seed": seed,
            "bootstrap_repetitions": bootstrap_reps,
            "bootstrap_seed": seed,
            "tfidf": TFIDF_SETTINGS,
            "text_preprocessing": "transductive without labels",
        },
        "predictors": {},
        "elapsed_seconds": None,
    }
    output.mkdir(parents=True, exist_ok=True)
    frozen = {
        "protocol": judge["manifest"],
        "source_n": len(y),
        "source_groups": len(set(groups)),
        "predictors": {},
    }
    validation_folds = make_grouped_folds(rows, 3, seed)
    for name, values in predictions.items():
        crossing = recall_crossing(y, values, ids)
        results["predictors"][name] = {
            "metrics": {**score(y, values), "ece": ece(y, values)},
            "bootstrap_95_ci": intervals["predictors"][name],
            "paired_probe_gain_95_ci": intervals["reference_minus_predictor"].get(name),
            "selected_outer_folds": choices[name],
            "source_budget_90": crossing,
            "budget_90_95_ci": budget_interval(y, values, ids, groups, bootstrap_reps, seed),
            "fixed_50_percent": check_budget(y, values, ids, 0.5),
        }
        if name not in ("probe", "output", "output_uncertainty", "input_only"):
            continue
        if name in ("probe", "input_only"):
            tensor = hidden if name == "probe" else input_hidden
            selected = select_final_hidden_hyperparameters(
                tensor, y, validation_folds, judge["layer_numbers"]
            )
            final_x = tensor[:, int(selected["layer_index"]), :]
        else:
            final_x = matrices[name]
            selected = {"lambda": select_lambda(final_x, y, validation_folds)}
        model = fit_logistic(final_x, y, float(selected["lambda"]))
        if not model.converged:
            raise RuntimeError(f"Final fit failed to converge: {label}/{name}")
        scaler = PlattScaler.fit(values, y)
        entry = {
            "selected": selected,
            "platt_from_source_oof": asdict(scaler),
            "source_budget_90": crossing["budget"] if crossing else None,
            "model_file": name + ".npz",
        }
        save_model(
            output / entry["model_file"],
            model,
            {
                "judge": label,
                "predictor": name,
                "selected": selected,
                "source_protocol": judge["manifest"],
                "target_fitting": False,
            },
        )
        frozen["predictors"][name] = entry
        results["predictors"][name]["selected_final_refit"] = selected
    write_json(output / "frozen.json", frozen)
    write_json(
        output / "folds.json",
        {
            "pair_to_fold": dict(zip(ids, outer)),
            "group_to_fold": dict(zip(groups.tolist(), outer)),
            "label_independent": True,
        },
    )
    write_jsonl(
        output / "predictions.jsonl",
        [
            {
                "pair_id": p,
                "group_id": str(groups[i]),
                "outer_fold": outer[i],
                "y": int(y[i]),
                "predictions": {name: float(v[i]) for name, v in predictions.items()},
            }
            for i, p in enumerate(ids)
        ],
    )
    results["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "results.json", results)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "judge",
                "predictor",
                "n",
                "questions",
                "reversals",
                "auroc",
                "brier",
                "ece",
                "budget_90",
            ]
        )
        for name, entry in results["predictors"].items():
            crossing = entry["source_budget_90"]
            writer.writerow(
                [
                    label,
                    name,
                    len(y),
                    len(set(groups)),
                    int(y.sum()),
                    entry["metrics"]["auroc"],
                    entry["metrics"]["brier"],
                    entry["metrics"]["ece"],
                    crossing["budget"] if crossing else "",
                ]
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="append", type=parse_spec, required=True)
    parser.add_argument("--embeddings", action="append", type=parse_spec, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--population", choices=("per-judge", "common"), default="per-judge")
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="Development/transfer subset: only three numeric baselines; not Table 1",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    specs, embeddings = unique_specs(args.judge), unique_specs(args.embeddings)
    common = None
    if args.population == "common":
        common = set.intersection(
            *(load_judge(path, include_hidden=False)["usable"] for path in specs.values())
        )
    results = {}
    for label, path in specs.items():
        results[label] = analyze(
            label,
            path,
            args.output_dir / label,
            embeddings.get(label),
            args.numeric_only,
            common,
            args.bootstrap_reps,
            args.seed,
        )
        print(
            f"{label}: {results[label]['n']} scored pairs; probe AUROC {results[label]['predictors']['probe']['metrics']['auroc']}"
        )
    write_json(args.output_dir / "results.json", {"judges": results, "results_recomputed": True})


if __name__ == "__main__":
    main()
