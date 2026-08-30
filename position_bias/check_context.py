from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from position_bias.io_utils import read_jsonl, write_json, write_jsonl
from position_bias.protocol import judge_messages

ALLOWANCES = (128, 256, 384, 512)


def parse_tokenizer_spec(spec: str) -> tuple[str, str, str]:
    if "=" not in spec or "@" not in spec:
        raise argparse.ArgumentTypeError("Tokenizer must be LABEL=MODEL_OR_PATH@REVISION")
    label, target = spec.split("=", maxsplit=1)
    model_or_path, revision = target.rsplit("@", maxsplit=1)
    if not label or not model_or_path or not revision:
        raise argparse.ArgumentTypeError("Tokenizer must be LABEL=MODEL_OR_PATH@REVISION")
    return label, model_or_path, revision


def render_length(tokenizer: Any, pair: dict[str, Any], order: str) -> int:
    token_ids = tokenizer.apply_chat_template(
        judge_messages(pair, order),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(token_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure context retention for a fixed evaluation dataset."
    )
    parser.add_argument("--input", type=Path, default=Path("data/judgebench.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/context_check"))
    parser.add_argument(
        "--tokenizer",
        action="append",
        required=True,
        type=parse_tokenizer_spec,
        metavar="LABEL=MODEL@REVISION",
        help="Repeat once per tokenizer family (Qwen3 and Llama-3.1 for the full check).",
    )
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--primary-allowance", type=int, default=256, choices=ALLOWANCES)
    parser.add_argument("--minimum-retention", type=float, default=0.60)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow pinned tokenizer code from the selected checkpoint repositories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    labels = [spec[0] for spec in args.tokenizer]
    if len(labels) != len(set(labels)):
        raise ValueError("Tokenizer labels must be unique")

    tokenizer_details: dict[str, Any] = {}
    lengths_by_pair: dict[str, dict[str, dict[str, int]]] = {row["pair_id"]: {} for row in rows}
    for label, model_or_path, revision in args.tokenizer:
        print(f"Loading tokenizer {label}: {model_or_path}@{revision}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_or_path,
            revision=revision,
            cache_dir=str(args.cache_dir),
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        x_ids = tokenizer.encode(" X", add_special_tokens=False)
        y_ids = tokenizer.encode(" Y", add_special_tokens=False)
        resolved_revision = (
            getattr(tokenizer, "_commit_hash", None)
            or tokenizer.init_kwargs.get("_commit_hash")
            or revision
        )
        tokenizer_details[label] = {
            "model_or_path": model_or_path,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "verdict_token_ids": {"X": x_ids, "Y": y_ids},
            "single_token_verdicts": len(x_ids) == 1 and len(y_ids) == 1,
        }
        for index, row in enumerate(rows, start=1):
            lengths_by_pair[row["pair_id"]][label] = {
                "AB": render_length(tokenizer, row, "AB"),
                "BA": render_length(tokenizer, row, "BA"),
            }
            if index % 100 == 0:
                print(f"  tokenized {index}/{len(rows)} pairs")

    annotated: list[dict[str, Any]] = []
    retention_counts = {str(allowance): 0 for allowance in ALLOWANCES}
    for row in rows:
        token_lengths = lengths_by_pair[row["pair_id"]]
        max_prompt_tokens = max(
            order_length
            for family_lengths in token_lengths.values()
            for order_length in family_lengths.values()
        )
        eligibility = {
            str(allowance): max_prompt_tokens + allowance <= args.context_limit
            for allowance in ALLOWANCES
        }
        for allowance, is_eligible in eligibility.items():
            retention_counts[allowance] += int(is_eligible)
        annotated.append(
            {
                **row,
                "prompt_tokens": token_lengths,
                "max_prompt_tokens": max_prompt_tokens,
                "context_eligible": eligibility,
            }
        )

    primary_key = str(args.primary_allowance)
    eligible = [row for row in annotated if row["context_eligible"][primary_key]]
    retention = len(eligible) / len(rows)
    gate_passed = retention >= args.minimum_retention
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "eligible.jsonl", eligible)
    write_jsonl(args.output_dir / "analysis.jsonl", eligible)
    report = {
        "input": str(args.input),
        "input_pairs": len(rows),
        "context_limit": args.context_limit,
        "primary_allowance": args.primary_allowance,
        "minimum_retention": args.minimum_retention,
        "retention_curve": {
            allowance: {
                "retained": count,
                "rate": count / len(rows),
            }
            for allowance, count in retention_counts.items()
        },
        "primary_retained": len(eligible),
        "primary_retention": retention,
        "retention_gate_passed": gate_passed,
        "analysis_size": len(eligible),
        "tokenizers": tokenizer_details,
    }
    write_json(args.output_dir / "report.json", report)
    status = "PASS" if gate_passed else "FAIL"
    print(
        f"Primary retention: {len(eligible)}/{len(rows)} ({retention:.1%}); "
        f"60% gate: {status}. Analysis set: {len(eligible)}."
    )
    if not gate_passed:
        raise SystemExit(2)
    if not all(details["single_token_verdicts"] for details in tokenizer_details.values()):
        raise SystemExit("X/Y are not single tokens for every supplied tokenizer")


if __name__ == "__main__":
    main()
