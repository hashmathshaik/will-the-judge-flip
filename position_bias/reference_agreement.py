from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from position_bias.io_utils import read_jsonl, write_json

DEFAULT_JUDGES = {
    "qwen3_1_7b": Path("runs/qwen3_1_7b_judgebench/judgments.jsonl"),
    "qwen3_4b": Path("runs/qwen3_4b_judgebench/judgments.jsonl"),
    "qwen3_8b": Path("runs/qwen3_8b_judgebench/judgments.jsonl"),
    "llama3_1_8b": Path("runs/llama3_1_8b_judgebench/judgments.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute JudgeBench reference agreement by position-flip status."
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.json"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/hf_cache"))
    parser.add_argument(
        "--common-intersection",
        type=Path,
        default=Path("runs/analysis_judgebench/common_intersection.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/analysis_judgebench/reference_agreement.json"),
    )
    return parser.parse_args()


def raw_arrow_paths(data_config: Path, cache_root: Path) -> list[Path]:
    config = json.loads(data_config.read_text(encoding="utf-8"))["judgebench"]
    revision = str(config["revision"])
    dataset_cache = cache_root / "ScalerLab___judge_bench" / "default" / "0.0.0" / revision
    paths = sorted(dataset_cache.glob("judge_bench-*.arrow"))
    if len(paths) != len(config["splits"]):
        raise FileNotFoundError(
            f"Expected {len(config['splits'])} JudgeBench Arrow files under "
            f"{dataset_cache}, found {len(paths)}"
        )
    return paths


def load_reference_labels(paths: list[Path]) -> dict[str, str]:
    labels: dict[str, str] = {}
    label_to_identity = {"A>B": "A", "B>A": "B"}
    for path in paths:
        for row in Dataset.from_file(str(path)):
            pair_id = str(row["pair_id"])
            raw_label = str(row["label"])
            if raw_label not in label_to_identity:
                raise ValueError(f"Unexpected JudgeBench label {raw_label!r}")
            identity = label_to_identity[raw_label]
            if pair_id in labels and labels[pair_id] != identity:
                raise ValueError(f"Conflicting reference labels for {pair_id}")
            labels[pair_id] = identity
    return labels


def summarize_judge(
    common_pair_ids: set[str],
    references: dict[str, str],
    judgments_path: Path,
) -> dict[str, Any]:
    judgments = {row["pair_id"]: row for row in read_jsonl(judgments_path)}
    missing = common_pair_ids - judgments.keys()
    if missing:
        raise ValueError(f"{judgments_path} is missing {len(missing)} common pairs")

    buckets = {
        "invariant": {"n": 0, "ab_correct": 0, "ba_correct": 0},
        "flipped": {"n": 0, "ab_correct": 0, "ba_correct": 0},
    }
    for pair_id in common_pair_ids:
        row = judgments[pair_id]
        ab = row["ab"].get("canonical_verdict")
        ba = row["ba"].get("canonical_verdict")
        if ab not in {"A", "B"} or ba not in {"A", "B"}:
            raise ValueError(f"Common pair {pair_id} lacks a canonical verdict")
        flipped = bool(row["y_pos"])
        if flipped != (ab != ba):
            raise ValueError(f"Stored flip label disagrees with verdicts for {pair_id}")

        bucket = buckets["flipped" if flipped else "invariant"]
        bucket["n"] += 1
        bucket["ab_correct"] += int(ab == references[pair_id])
        bucket["ba_correct"] += int(ba == references[pair_id])

    for bucket in buckets.values():
        n = bucket["n"]
        bucket["ab_accuracy"] = bucket["ab_correct"] / n
        bucket["ba_accuracy"] = bucket["ba_correct"] / n
        bucket["two_order_mean_accuracy"] = (bucket["ab_correct"] + bucket["ba_correct"]) / (2 * n)
    return buckets


def main() -> None:
    args = parse_args()
    common_rows = read_jsonl(args.common_intersection)
    common_pair_ids = {str(row["pair_id"]) for row in common_rows}
    if len(common_pair_ids) != len(common_rows):
        raise ValueError("Common intersection contains duplicate pair IDs")

    references = load_reference_labels(raw_arrow_paths(args.data_config, args.cache_root))
    missing_references = common_pair_ids - references.keys()
    if missing_references:
        raise ValueError(f"Reference labels are missing for {len(missing_references)} common pairs")

    results = {
        "analysis": "post-hoc descriptive reference agreement",
        "comparison": "canonical verdict against the binary JudgeBench preference",
        "common_n": len(common_pair_ids),
        "judges": {
            name: summarize_judge(common_pair_ids, references, path)
            for name, path in DEFAULT_JUDGES.items()
        },
    }
    write_json(args.output, results)
    print(f"Wrote {args.output}")
    for name, result in results["judges"].items():
        stable = result["invariant"]["ab_accuracy"]
        flipped = result["flipped"]["ab_accuracy"]
        print(f"{name}: invariant AB={stable:.3f}; flipped AB={flipped:.3f}")


if __name__ == "__main__":
    main()
