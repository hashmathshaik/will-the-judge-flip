from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from position_bias.artifacts import aligned, load_judge, load_model, selected_rows
from position_bias.features import BASELINE_NAMES, numeric_baselines
from position_bias.io_utils import read_jsonl, write_json
from position_bias.metrics import score
from position_bias.protocol import PROTOCOL_VERSION, SYSTEM_PROMPT, USER_TEMPLATE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.pilot_dir
    selection = json.loads((root / "selection.json").read_text())
    model = selection["model"]
    judge_name = selection["judge"]
    report = {
        "status": "passed",
        "checkpoint": model,
        "benchmarks": {},
        "evaluation": "question-grouped source fitting and frozen target transfer",
    }
    for benchmark, selected in selection["benchmarks"].items():
        run = load_judge(root / f"{judge_name}_{benchmark}")
        manifest = run["manifest"]
        if (
            manifest["model"] != model["model_id"]
            or manifest["requested_revision"] != model["revision"]
        ):
            raise ValueError("Pilot used a different checkpoint than its selection record")
        if len(run["records"]) != selected["selected"]:
            raise ValueError(f"Incomplete real run: {benchmark}")
        if set(run["records"]) != set(selected["pair_ids"]):
            raise ValueError(f"Generated pairs differ from the selected cohort: {benchmark}")
        if (
            manifest["protocol_version"] != PROTOCOL_VERSION
            or manifest["system_prompt"] != SYSTEM_PROMPT
            or manifest["user_template"] != USER_TEMPLATE
        ):
            raise ValueError("Pilot generations used a different prompt or parser protocol")
        expected_samples = 0 if benchmark == "mtbench" else 5
        for record in run["records"].values():
            if record.get("exclusion"):
                continue
            if len(record.get("samples", [])) != expected_samples:
                raise ValueError("Actual extra AB call count differs from the pilot configuration")
            ab, ba = record["ab"]["parse"], record["ba"]["parse"]
            expected_y = (
                int(ab["slot"] == ba["slot"]) if ab["format_ok"] and ba["format_ok"] else None
            )
            if record["y_pos"] != expected_y:
                raise ValueError("Saved reversal label disagrees with the displayed AB/BA slots")
        rows = selected_rows(run)
        report["benchmarks"][benchmark] = {
            "attempted": len(run["records"]),
            "retained": len(rows),
            "reversals": sum(r["y_pos"] for r in rows),
            "device": manifest["device"],
            "dtype": manifest["dtype"],
            "gpu_name": manifest.get("gpu_name"),
            "protocol_version": manifest["protocol_version"],
            "elapsed_seconds": manifest["elapsed_seconds"],
            "ab_parse_valid": manifest["summary"]["orders"]["AB"]["format_ok"],
            "ba_parse_valid": manifest["summary"]["orders"]["BA"]["format_ok"],
            "extraction_errors": manifest["summary"]["extraction_errors"],
            "run_manifest_sha256": hashlib.sha256(
                (root / f"{judge_name}_{benchmark}" / "run.json").read_bytes()
            ).hexdigest(),
        }
        if benchmark != "mtbench":
            result = json.loads(
                (root / f"analysis_{benchmark}" / judge_name / "results.json").read_text()
            )
            if not result["full_ten_baselines"] or not set(BASELINE_NAMES).issubset(
                result["predictors"]
            ):
                raise ValueError(f"Incomplete baseline comparison: {benchmark}")
            predictions = read_jsonl(
                root / f"analysis_{benchmark}" / judge_name / "predictions.jsonl"
            )
            folds = {r["group_id"]: set() for r in predictions}
            for row in predictions:
                folds[row["group_id"]].add(row["outer_fold"])
            if (
                any(len(values) != 1 for values in folds.values())
                or len({r["outer_fold"] for r in predictions}) != 5
            ):
                raise ValueError("Question-grouped outer folds were not preserved")
            y = np.asarray([r["y"] for r in predictions])
            for predictor, entry in result["predictors"].items():
                values = np.asarray([r["predictions"][predictor] for r in predictions])
                actual = score(y, values)
                if actual["auroc"] != entry["metrics"]["auroc"]:
                    raise ValueError("Saved AUROC disagrees with held-out predictions")
            report["benchmarks"][benchmark]["auroc"] = {
                name: entry["metrics"]["auroc"] for name, entry in result["predictors"].items()
            }

    source = root / "analysis_judgebench" / judge_name
    frozen = json.loads((source / "frozen.json").read_text())
    target = load_judge(root / f"{judge_name}_mtbench")
    rows = selected_rows(target)
    hidden, y, metadata = aligned(target, rows)
    matrices = numeric_baselines(metadata, rows)
    saved = {
        r["pair_id"]: r
        for r in read_jsonl(root / "transfer_mtbench" / judge_name / "predictions.jsonl")
    }
    if set(saved) != {r["pair_id"] for r in rows}:
        raise ValueError("Frozen target predictions do not cover every scored target pair")
    for name, entry in frozen["predictors"].items():
        features = (
            hidden[:, int(entry["selected"]["layer_index"]), :]
            if name == "probe"
            else (
                np.stack([target["input_hidden_by_id"][r["pair_id"]] for r in rows])[
                    :, int(entry["selected"]["layer_index"]), :
                ]
                if name == "input_only"
                else matrices[name]
            )
        )
        predictions = load_model(source / entry["model_file"]).predict(features)
        np.testing.assert_allclose(
            predictions,
            [saved[r["pair_id"]]["predictions"][name] for r in rows],
            atol=1e-12,
            rtol=1e-12,
        )
    transfer = json.loads((root / "transfer_mtbench" / judge_name / "results.json").read_text())
    report["benchmarks"]["mtbench"].update(
        {
            "auroc": {
                name: result["metrics"]["auroc"] for name, result in transfer["predictors"].items()
            },
            "probe_fixed_50_percent": transfer["predictors"]["probe"]["fixed_50_percent"],
            "source_chosen_budget": transfer["predictors"]["probe"]["source_chosen_budget"],
        }
    )
    report["frozen_transfer_independently_recomputed"] = True
    write_json(root / "pilot-report.json", report)
    lines = [
        f"# Real {model['model_id']} pilot",
        "",
        "All runs use actual pinned model weights and real benchmark inputs.",
        "",
        "| Benchmark | Attempted | Retained | Reversals | Probe AUROC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["benchmarks"].items():
        value = result["auroc"]["probe"]
        lines.append(
            f"| {name} | {result['attempted']} | {result['retained']} | {result['reversals']} | {value:.3f} |"
            if value is not None
            else f"| {name} | {result['attempted']} | {result['retained']} | {result['reversals']} | Not estimable |"
        )
    (root / "PILOT_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
