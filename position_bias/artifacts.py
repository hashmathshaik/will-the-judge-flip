from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from position_bias.io_utils import read_jsonl, write_json
from position_bias.probe import LogisticModel, Standardizer
from position_bias.protocol import PROTOCOL_VERSION, canonicalize


def parse_spec(spec: str) -> tuple[str, Path]:
    label, separator, path = spec.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    return label, Path(path)


def unique_specs(specs: list[tuple[str, Path]]) -> dict[str, Path]:
    result = dict(specs)
    if len(result) != len(specs):
        raise ValueError("Duplicate judge labels")
    return result


def load_judge(directory: Path, include_hidden: bool = True) -> dict:
    manifest = json.loads((directory / "run.json").read_text())
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"{directory} uses a different protocol; supply a run matching the configured protocol"
        )
    raw = read_jsonl(directory / "judgments.jsonl")
    metadata_rows = read_jsonl(directory / "feature_metadata.jsonl")
    records = {r["pair_id"]: r for r in raw}
    metadata = {r["pair_id"]: r for r in metadata_rows}
    if len(records) != len(raw) or len(metadata) != len(metadata_rows):
        raise ValueError(f"Duplicate pair IDs in {directory}")
    with np.load(directory / "features.npz", allow_pickle=False) as data:
        ids = [str(i) for i in data["pair_ids"]]
        layers = [int(i) for i in data["layer_numbers"]]
        hidden = data["hidden_states"] if include_hidden else None
        input_hidden = (
            data["input_hidden_states"]
            if include_hidden and "input_hidden_states" in data
            else None
        )
    if len(ids) != len(set(ids)) or layers != manifest["candidate_layers"]:
        raise ValueError("Duplicate feature IDs or inconsistent candidate layers")
    if hidden is not None and (
        len(hidden) != len(ids)
        or hidden.ndim != 3
        or hidden.shape[1] != len(layers)
        or not np.isfinite(hidden).all()
    ):
        raise ValueError("Missing, malformed, or nonfinite hidden-state features")
    usable = set()
    for pair in set(ids) & set(metadata) & set(records):
        r = records[pair]
        if not (r["ab"]["parse"]["format_ok"] and r["ba"]["parse"]["format_ok"]):
            continue
        ab_choice = canonicalize(r["ab"]["parse"]["slot"], "AB")
        ba_choice = canonicalize(r["ba"]["parse"]["slot"], "BA")
        if r["ab"]["canonical_verdict"] != ab_choice or r["ba"]["canonical_verdict"] != ba_choice:
            raise ValueError(
                f"Displayed slots were incorrectly mapped to response identity for {pair}"
            )
        expected = int(ab_choice != ba_choice)
        if r["y_pos"] != expected:
            raise ValueError(f"Incorrect canonical reversal label for {pair}")
        usable.add(pair)
    return {
        "manifest": manifest,
        "records": records,
        "metadata": metadata,
        "usable": usable,
        "layer_numbers": layers,
        "feature_ids": ids,
        "hidden_by_id": {p: hidden[i] for i, p in enumerate(ids)} if hidden is not None else {},
        "input_hidden_by_id": {p: input_hidden[i] for i, p in enumerate(ids)}
        if input_hidden is not None
        else {},
        "run_directory": str(directory),
    }


def selected_rows(judge: dict, ids: set[str] | None = None) -> list[dict]:
    return [
        judge["records"][p]
        for p in sorted(judge["usable"] if ids is None else judge["usable"] & ids)
    ]


def coverage(judge: dict, rows: list[dict]) -> dict:
    selected = {r["pair_id"] for r in rows}
    counts = {
        "context_limit": 0,
        "malformed_ab": 0,
        "malformed_ba": 0,
        "missing_activation_or_metadata": 0,
        "outside_common_population": 0,
    }
    excluded = []
    for pair_id, record in judge["records"].items():
        if pair_id in selected:
            continue
        if record.get("exclusion"):
            reason = "context_limit"
        elif not record["ab"]["parse"]["format_ok"]:
            reason = "malformed_ab"
        elif not record["ba"]["parse"]["format_ok"]:
            reason = "malformed_ba"
        elif pair_id not in judge["usable"]:
            reason = "missing_activation_or_metadata"
        else:
            reason = "outside_common_population"
        counts[reason] += 1
        excluded.append({"pair_id": pair_id, "reason": reason})
    return {
        "attempted": len(judge["records"]),
        "retained": len(rows),
        "retained_fraction": len(rows) / len(judge["records"]) if judge["records"] else None,
        "excluded_counts": counts,
        "excluded_pairs": excluded,
    }


def aligned(judge: dict, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if not rows:
        raise ValueError("No scored pairs")
    return (
        np.stack([judge["hidden_by_id"][r["pair_id"]] for r in rows]),
        np.asarray([r["y_pos"] for r in rows], dtype=int),
        [judge["metadata"][r["pair_id"]] for r in rows],
    )


def check_frozen_protocol(source: dict, target: dict) -> None:
    for key in (
        "protocol_version",
        "model",
        "requested_revision",
        "system_prompt",
        "user_template",
        "max_new_tokens",
        "do_sample",
        "qwen_thinking",
        "candidate_layers",
        "dtype",
        "context_limit",
    ):
        if key not in source or source[key] != target.get(key):
            raise ValueError(f"Frozen transfer mismatch: {key}")
    for key in (
        "resolved_model_revision",
        "resolved_tokenizer_revision",
        "verdict_token_ids",
        "feature_capture",
        "package_versions",
    ):
        if source.get(key) != target.get(key):
            raise ValueError(f"Frozen transfer mismatch: {key}")


def save_model(path: Path, model: LogisticModel, provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights=model.weights if model.weights is not None else np.empty(0),
        intercept=np.asarray(model.intercept),
        constant=np.asarray(model.constant if model.constant is not None else np.nan),
        mean=model.standardizer.mean,
        scale=model.standardizer.scale,
        lambda_value=np.asarray(model.lambda_value),
    )
    write_json(path.with_suffix(".json"), provenance)


def load_model(path: Path) -> LogisticModel:
    with np.load(path, allow_pickle=False) as data:
        weights = data["weights"].copy()
        constant = float(data["constant"])
        return LogisticModel(
            weights if len(weights) else None,
            float(data["intercept"]),
            None if np.isnan(constant) else constant,
            Standardizer(data["mean"].copy(), data["scale"].copy()),
            float(data["lambda_value"]),
            True,
        )
