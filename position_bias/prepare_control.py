from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from position_bias.io_utils import read_jsonl, write_json, write_jsonl

DEFAULT_SEED = 20260828
DEFAULT_SIZE = 100


def _unique_by_id(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id in by_id:
            raise ValueError(f"Duplicate pair_id {pair_id!r} in {source}")
        by_id[pair_id] = row
    return by_id


def selection_key(pair_id: str, seed: int) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).digest()
    return digest, pair_id


def select_control_rows(
    dataset_rows: list[dict[str, Any]],
    common_rows: list[dict[str, Any]],
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    dataset_by_id = _unique_by_id(dataset_rows, "dataset")
    common_by_id = _unique_by_id(common_rows, "common intersection")
    if size < 1 or size > len(common_by_id):
        raise ValueError(f"Control size must be between 1 and {len(common_by_id)}, got {size}")
    missing = sorted(set(common_by_id) - set(dataset_by_id))
    if missing:
        raise ValueError(f"{len(missing)} common pairs are missing from the dataset")

    selected_ids = sorted(common_by_id, key=lambda pair_id: selection_key(pair_id, seed))[:size]
    return [dataset_by_id[pair_id] for pair_id in selected_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the shared JudgeBench subset for the repeated-AB control."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--common-intersection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_rows = read_jsonl(args.dataset)
    common_rows = read_jsonl(args.common_intersection)
    selected = select_control_rows(dataset_rows, common_rows, args.size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = args.output_dir / "subset.jsonl"
    write_jsonl(subset_path, selected)
    write_json(
        args.output_dir / "subset_manifest.json",
        {
            "common_intersection": str(args.common_intersection),
            "common_n": len(common_rows),
            "dataset": str(args.dataset),
            "pair_ids": [row["pair_id"] for row in selected],
            "seed": args.seed,
            "selection": "first pair IDs ranked by SHA-256(seed:pair_id), then pair_id",
            "subset_n": len(selected),
        },
    )
    print(f"Frozen control subset: {len(selected)}/{len(common_rows)} pairs")
    print(f"Wrote {subset_path}")


if __name__ == "__main__":
    main()
