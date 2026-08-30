from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from datasets import load_dataset

from position_bias.io_utils import write_json, write_jsonl

DATASET_ID = "lmsys/mt_bench_human_judgments"
DEFAULT_REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
SPLITS = ("gpt4_pair", "human")


def role_contents(conversation: list[dict[str, str]], role: str) -> list[str]:
    return [str(message["content"]) for message in conversation if message["role"] == role]


def serialize_comparison_side(
    conversation: list[dict[str, str]], turn: int
) -> tuple[list[str], str]:
    user_turns = role_contents(conversation, "user")
    assistant_turns = role_contents(conversation, "assistant")
    if turn not in (1, 2):
        raise ValueError(f"Unexpected MT-Bench turn: {turn}")
    if len(user_turns) < turn or len(assistant_turns) < turn:
        raise ValueError("Conversation is shorter than its declared turn")
    if turn == 1:
        return user_turns[:1], assistant_turns[0]
    candidate = (
        f"ASSISTANT TURN 1:\n{assistant_turns[0]}\n\nASSISTANT TURN 2:\n{assistant_turns[1]}"
    )
    return user_turns[:2], candidate


def serialize_user_request(user_turns: list[str]) -> str:
    if len(user_turns) == 1:
        return user_turns[0]
    return (
        "This is a two-turn conversation. Evaluate each candidate's second-turn "
        "answer in light of its own first-turn answer, which is included in that "
        "candidate.\n\n"
        f"USER TURN 1:\n{user_turns[0]}\n\n"
        f"USER TURN 2:\n{user_turns[1]}"
    )


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    turn = int(row["turn"])
    users_a, candidate_a = serialize_comparison_side(row["conversation_a"], turn)
    users_b, candidate_b = serialize_comparison_side(row["conversation_b"], turn)
    if users_a != users_b:
        raise ValueError(
            f"Model conversations disagree on user turns for question {row['question_id']}"
        )

    sides = sorted(
        [
            (str(row["model_a"]), candidate_a),
            (str(row["model_b"]), candidate_b),
        ],
        key=lambda side: (side[0], side[1]),
    )
    identity = json.dumps(
        {
            "question_id": int(row["question_id"]),
            "turn": turn,
            "sides": sides,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    pair_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
    return {
        "pair_id": pair_id,
        "group_id": f"mt-bench::{int(row['question_id'])}",
        "source": "mt-bench",
        "original_id": str(row["question_id"]),
        "coarse_domain": "mt-bench",
        "turn": turn,
        "model_a": sides[0][0],
        "model_b": sides[1][0],
        "question": serialize_user_request(users_a),
        "response_a": sides[0][1],
        "response_b": sides[1][1],
    }


def stable_rank(pair_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize unique MT-Bench comparisons without preference labels."
    )
    parser.add_argument("--output", type=Path, default=Path("data/mtbench.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/mtbench_prepare_report.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    dataset = load_dataset(
        DATASET_ID,
        revision=args.revision,
        cache_dir=str(args.cache_dir),
    )
    raw_counts = {split: len(dataset[split]) for split in SPLITS}
    unique: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        for raw_row in dataset[split]:
            row = normalize_row(dict(raw_row))
            unique.setdefault(row["pair_id"], row)
    rows = sorted(unique.values(), key=lambda row: stable_rank(row["pair_id"], args.seed))
    selected = rows if args.limit is None else rows[: args.limit]
    write_jsonl(args.output, selected)
    write_json(
        args.report,
        {
            "dataset_id": DATASET_ID,
            "dataset_revision": args.revision,
            "raw_counts": raw_counts,
            "raw_total": sum(raw_counts.values()),
            "annotation_duplicates_removed": sum(raw_counts.values()) - len(rows),
            "unique_comparisons": len(rows),
            "unique_questions": len({row["group_id"] for row in rows}),
            "written_total": len(selected),
            "development_limit": args.limit,
            "human_preference_labels_used": False,
            "seed": args.seed,
            "output": str(args.output),
        },
    )
    print(
        f"Wrote {len(selected)} unique MT-Bench comparisons from "
        f"{sum(raw_counts.values())} annotation rows."
    )


if __name__ == "__main__":
    main()
