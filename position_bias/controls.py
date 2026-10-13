from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from position_bias.artifacts import aligned, load_judge, selected_rows
from position_bias.audit import recall_crossing
from position_bias.features import all_baselines
from position_bias.folds import make_grouped_folds
from position_bias.io_utils import read_jsonl, write_json, write_jsonl
from position_bias.metrics import bootstrap_prediction_suite, score
from position_bias.probe import (
    LAMBDA_GRID,
    Standardizer,
    brier_score,
    fit_logistic,
    nested_baseline_predict,
    nested_predict,
)


def adjustment_predictions(
    hidden: np.ndarray,
    visible: np.ndarray | None,
    labels: np.ndarray,
    rows: list[dict],
    outer: list[int],
    layer_indices: dict[int, int],
    seed: int = 42,
    dimension: int = 30,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(hidden.shape[-1], dimension)) / np.sqrt(dimension)
    result = np.full(len(rows), np.nan)

    def residualize(train, evaluate, x):
        if visible is None:
            return x[train], x[evaluate]
        scaler = Standardizer.fit(visible[train])
        ridge = Ridge(alpha=1.0).fit(scaler.transform(visible[train]), x[train])
        return x[train] - ridge.predict(scaler.transform(visible[train])), x[
            evaluate
        ] - ridge.predict(scaler.transform(visible[evaluate]))

    for fold in sorted(set(outer)):
        train = np.flatnonzero(np.asarray(outer) != fold)
        held_out = np.flatnonzero(np.asarray(outer) == fold)
        x = hidden[:, layer_indices[fold], :] @ projection
        inner = make_grouped_folds([rows[i] for i in train], 3, seed + 1000 + fold)
        scores = []
        for penalty in LAMBDA_GRID:
            errors = []
            for inner_fold in sorted(set(inner)):
                fit_ids = train[np.asarray(inner) != inner_fold]
                val_ids = train[np.asarray(inner) == inner_fold]
                fit_x, val_x = residualize(fit_ids, val_ids, x)
                model = fit_logistic(fit_x, labels[fit_ids], penalty)
                errors.append(brier_score(labels[val_ids], model.predict(val_x)))
            scores.append((float(np.mean(errors)), penalty))
        penalty = min(scores)[1]
        train_x, eval_x = residualize(train, held_out, x)
        result[held_out] = fit_logistic(train_x, labels[train], penalty).predict(eval_x)
    if not np.isfinite(result).all():
        raise RuntimeError("Control did not produce a held-out prediction for each pair")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.analysis / "results.json").read_text())
    if source["population"] != "per-judge-retained":
        raise ValueError("Control currently expects the primary per-judge population")
    judge = load_judge(args.run)
    rows = selected_rows(judge)
    hidden, y, metadata = aligned(judge, rows)
    baseline = all_baselines(metadata, rows, args.embeddings)
    assignments = json.loads((args.analysis / "folds.json").read_text())["pair_to_fold"]
    outer = [assignments[r["pair_id"]] for r in rows]
    layers = {
        entry["outer_fold"]: entry["layer_index"]
        for entry in source["predictors"]["probe"]["selected_outer_folds"]
    }
    visible = {
        "unadjusted_projection": None,
        "length": baseline["output"][:, 3:8],
        "choice": baseline["output"][:, -1:],
        "margin": baseline["output"][:, 1:3],
        "all_visible": baseline["all_visible"],
    }
    predictions = {
        name: adjustment_predictions(
            hidden, x, y, rows, outer, layers, seed=source["configuration"]["seed"]
        )
        for name, x in visible.items()
    }
    source_predictions = {
        r["pair_id"]: r["predictions"]["probe"]
        for r in read_jsonl(args.analysis / "predictions.jsonl")
    }
    predictions["full_probe"] = np.asarray([source_predictions[r["pair_id"]] for r in rows])

    def inner_builder(train_rows, fold):
        return make_grouped_folds(train_rows, 3, source["configuration"]["seed"] + 1000 + fold)

    predictions["margin_only"], margin_choices = nested_baseline_predict(
        baseline["output"][:, 1:3], y, outer, inner_builder, rows
    )
    depth_choices = {}
    for index, layer in enumerate(judge["layer_numbers"]):
        name = f"depth_{layer}"
        predictions[name], depth_choices[name] = nested_predict(
            hidden[:, index : index + 1, :], y, outer, inner_builder, rows, layer_numbers=[layer]
        )
    groups = np.asarray([row["group_id"] for row in rows])
    intervals = bootstrap_prediction_suite(
        y,
        predictions,
        groups,
        reference_name="full_probe",
        repetitions=source["configuration"]["bootstrap_repetitions"],
        seed=source["configuration"]["seed"],
    )
    write_json(
        args.output_dir / "results.json",
        {
            "n": len(rows),
            "dimension": 30,
            "ridge_alpha": 1.0,
            "projection": "seeded Gaussian",
            "layer_selection": "full probe selection within each outer training fold",
            "nuisance_fitting": "training only in every inner and outer fold",
            "causal_claim": False,
            "metrics": {name: score(y, p) for name, p in predictions.items()},
            "bootstrap": intervals,
            "budgets_90": {
                name: recall_crossing(y, p, [r["pair_id"] for r in rows])
                for name, p in predictions.items()
            },
            "margin_choices": margin_choices,
            "per_depth_choices": depth_choices,
        },
    )
    write_jsonl(
        args.output_dir / "predictions.jsonl",
        [
            {
                "pair_id": r["pair_id"],
                "y": int(y[i]),
                "predictions": {n: float(v[i]) for n, v in predictions.items()},
            }
            for i, r in enumerate(rows)
        ],
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["length", "choice", "margin", "all_visible"]
    x = np.arange(len(names))
    full = score(y, predictions["full_probe"])["auroc"]
    projected = score(y, predictions["unadjusted_projection"])["auroc"]
    residual = [score(y, predictions[name])["auroc"] for name in names]
    figure, axis = plt.subplots(figsize=(7, 3.8))
    axis.bar(x - 0.25, [full] * len(names), width=0.25, color="gray", label="Full probe")
    axis.bar(
        x, [projected] * len(names), width=0.25, color="#54a76d", label="30-dimensional projection"
    )
    axis.bar(x + 0.25, residual, width=0.25, color="#4d8ccc", label="30-dimensional residual")
    axis.set(xticks=x, xticklabels=names, ylabel="Held-out AUROC", ylim=(0, 1))
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "activation_controls.png", dpi=180)
    figure.savefig(args.output_dir / "activation_controls.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
