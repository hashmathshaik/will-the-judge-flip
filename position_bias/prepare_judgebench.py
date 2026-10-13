from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from position_bias.io_utils import write_json, write_jsonl
from position_bias.protocol import coarse_domain

DATASET_ID = "ScalerLab/JudgeBench"
DEFAULT_REVISION = "57dd5e0b9817d07f05ec8f45a91b2ce1e310e308"
SPLITS = ("gpt", "claude")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_group_id(source: str, original_id: Any, question: str) -> str:
    if original_id is None:
        digest = hashlib.sha1(normalize_whitespace(question).encode("utf-8")).hexdigest()
        return f"{source}::q{digest[:12]}"
    return f"{source}::{original_id}"


def normalize_row(row: dict[str, Any], generator_split: str) -> dict[str, Any]:
    source = str(row["source"])
    original_id = row["original_id"]
    return {
        "pair_id": str(row["pair_id"]),
        "group_id": build_group_id(source, original_id, str(row["question"])),
        "generator_split": generator_split,
        "source": source,
        "original_id": None if original_id is None else str(original_id),
        "coarse_domain": coarse_domain(source),
        "question": str(row["question"]),
        "response_a": str(row["response_A"]),
        "response_b": str(row["response_B"]),
    }


def exact_pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    responses = sorted((row["response_a"].strip(), row["response_b"].strip()))
    return row["question"].strip(), responses[0], responses[1]


def stable_rank(pair_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, normalize, and deduplicate the JudgeBench splits."
    )
    parser.add_argument("--output", type=Path, default=Path("data/judgebench.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/judgebench_prepare_report.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Deterministic development subset. Omit for the full upstream materialization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    normalized: list[dict[str, Any]] = []
    raw_counts: dict[str, int] = {}
    for split in SPLITS:
        dataset = load_dataset(
            DATASET_ID,
            split=split,
            revision=args.revision,
            cache_dir=str(args.cache_dir),
        )
        raw_counts[split] = len(dataset)
        normalized.extend(normalize_row(dict(row), split) for row in dataset)

    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in normalized:
        key = exact_pair_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    unique.sort(key=lambda row: stable_rank(row["pair_id"], args.seed))
    selected = unique if args.limit is None else unique[: args.limit]
    write_jsonl(args.output, selected)

    report = {
        "dataset_id": DATASET_ID,
        "dataset_revision": args.revision,
        "raw_counts": raw_counts,
        "raw_total": len(normalized),
        "exact_duplicates_removed": len(normalized) - len(unique),
        "unique_total": len(unique),
        "written_total": len(selected),
        "development_limit": args.limit,
        "seed": args.seed,
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(
        f"Wrote {len(selected)} pairs to {args.output} "
        f"({len(normalized) - len(unique)} exact duplicates removed)."
    )


if __name__ == "__main__":
    main()
