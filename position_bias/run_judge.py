from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from position_bias.io_utils import append_jsonl, read_jsonl, write_json, write_jsonl
from position_bias.modeling import (
    GeneratedJudgment,
    generate_judgment,
    load_runtime,
    teacher_forced_extract,
)
from position_bias.protocol import (
    PROTOCOL_VERSION,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    build_user_prompt,
    canonicalize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic AB/BA judgments and extract pre-verdict features."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Pinned Hub model ID or snapshot path")
    parser.add_argument("--revision", required=True, help="Exact 40-character Hub commit")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", default="auto", choices=("auto", "float32", "bfloat16", "float16")
    )
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        choices=(0, 5),
        help="Five extra stochastic AB calls for sampling baselines; 0 for transfer-only runs",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow pinned model/tokenizer code from the checkpoint repository.",
    )
    parser.add_argument(
        "--extract-features",
        action="store_true",
        help="Extract pre-verdict hidden states, logits, and baseline features.",
    )
    parser.add_argument(
        "--repeat-ab",
        action="store_true",
        help="Also run identical AB2 as the predeclared nondeterminism control.",
    )
    return parser.parse_args()


def judgment_dict(judgment: GeneratedJudgment, order: str) -> dict[str, Any]:
    parsed = judgment.parsed
    canonical = canonicalize(parsed.slot, order) if parsed.format_ok and parsed.slot else None
    return {
        "order": order,
        "prompt_ids": judgment.prompt_ids,
        "completion_ids": judgment.completion_ids,
        "completion": judgment.completion,
        "prompt_tokens": len(judgment.prompt_ids),
        "completion_tokens": len(judgment.completion_ids),
        "stopped_on_eos": judgment.stopped_on_eos,
        "parse": parsed.to_dict(),
        "canonical_verdict": canonical,
        "feature_error": judgment.feature_error,
    }


def summarize(records: list[dict[str, Any]], orders: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"pairs": len(records), "orders": {}}
    for order in orders:
        order_rows = [record[order.lower()] for record in records if order.lower() in record]
        summary["orders"][order] = {
            "format_ok": sum(row["parse"]["format_ok"] for row in order_rows),
            "format_rate": sum(row["parse"]["format_ok"] for row in order_rows) / len(order_rows)
            if order_rows
            else None,
            "confidence_ok": sum(row["parse"]["confidence_ok"] for row in order_rows),
            "confidence_rate": sum(row["parse"]["confidence_ok"] for row in order_rows)
            / len(order_rows)
            if order_rows
            else None,
            "verdict_reached": sum(row["parse"]["verdict_reached"] for row in order_rows),
            "verdict_reached_rate": sum(row["parse"]["verdict_reached"] for row in order_rows)
            / len(order_rows)
            if order_rows
            else None,
        }

    usable = [
        record
        for record in records
        if record["ab"]["parse"]["format_ok"] and record["ba"]["parse"]["format_ok"]
    ]
    flips = sum(record["y_pos"] == 1 for record in usable)
    summary["usable_ab_ba"] = len(usable)
    summary["position_flips"] = flips
    summary["pfr"] = flips / len(usable) if usable else None
    summary["pfr_degenerate"] = summary["pfr"] is not None and (
        summary["pfr"] < 0.05 or summary["pfr"] > 0.95
    )
    if "AB2" in orders:
        repeat_usable = [
            record
            for record in records
            if record["ab"]["parse"]["format_ok"]
            and record.get("ab2", {}).get("parse", {}).get("format_ok", False)
        ]
        disagreements = sum(
            record["ab"]["canonical_verdict"] != record["ab2"]["canonical_verdict"]
            for record in repeat_usable
        )
        summary["same_order_control"] = {
            "usable": len(repeat_usable),
            "disagreements": disagreements,
            "rate": disagreements / len(repeat_usable) if repeat_usable else None,
        }
    return summary


def main() -> None:
    args = parse_args()
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        raise ValueError("--revision must be an exact lowercase 40-character commit")
    rows = read_jsonl(args.input)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("The selected input is empty")

    if args.max_new_tokens < 1 or args.context_limit < 1 or args.temperature <= 0:
        raise ValueError("Token limits and sampling temperature must be positive")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("Input contains duplicate pair IDs")
    if (args.output_dir / "judgments.jsonl").exists():
        raise FileExistsError("Use a fresh output directory; overwriting judgments is disabled")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Loading {args.model}@{args.revision} for {len(rows)} pairs...")
    runtime = load_runtime(
        model_or_path=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        requested_device=args.device,
        requested_dtype=args.dtype,
        local_files_only=args.local_files_only,
        cpu_threads=args.cpu_threads,
        trust_remote_code=args.trust_remote_code,
    )
    print(
        f"Loaded on {runtime.device} as {runtime.dtype}; candidate layers "
        f"{runtime.layer_numbers}; verdict tokens {runtime.verdict_token_ids}."
    )

    records: list[dict[str, Any]] = []
    generation_cache: dict[str, GeneratedJudgment] = {}
    for index, pair in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {pair['pair_id']}: AB", flush=True)
        try:
            ab = generate_judgment(
                runtime,
                pair,
                "AB",
                args.max_new_tokens,
                args.context_limit,
                capture_features=args.extract_features,
            )
            ba = generate_judgment(runtime, pair, "BA", args.max_new_tokens, args.context_limit)
        except ValueError as exc:
            if "above the frozen" not in str(exc):
                raise
            failed = {"parse": {"format_ok": False}, "canonical_verdict": None}
            records.append(
                {
                    "pair_id": pair["pair_id"],
                    "group_id": pair["group_id"],
                    "source": pair["source"],
                    "coarse_domain": pair["coarse_domain"],
                    "ab": {
                        **failed,
                        "parse": {
                            "format_ok": False,
                            "confidence_ok": False,
                            "verdict_reached": False,
                        },
                    },
                    "ba": {
                        **failed,
                        "parse": {
                            "format_ok": False,
                            "confidence_ok": False,
                            "verdict_reached": False,
                        },
                    },
                    "y_pos": None,
                    "exclusion": str(exc),
                }
            )
            append_jsonl(args.output_dir / "judgments.jsonl", records[-1])
            continue
        print(f"[{index}/{len(rows)}] {pair['pair_id']}: BA", flush=True)
        record: dict[str, Any] = {
            "pair_id": pair["pair_id"],
            "group_id": pair["group_id"],
            "source": pair["source"],
            "coarse_domain": pair["coarse_domain"],
            "ab": judgment_dict(ab, "AB"),
            "ba": judgment_dict(ba, "BA"),
            "input_text": build_user_prompt(pair, "AB"),
        }
        if args.samples:
            sample_seed = int.from_bytes(
                hashlib.sha256(f"{args.seed}:{pair['pair_id']}".encode()).digest()[:4], "little"
            )
            torch.manual_seed(sample_seed)
            record["samples"] = [
                judgment_dict(
                    generate_judgment(
                        runtime,
                        pair,
                        "AB",
                        args.max_new_tokens,
                        args.context_limit,
                        sample=True,
                        temperature=args.temperature,
                    ),
                    "AB",
                )
                for _ in range(args.samples)
            ]
        if ab.parsed.format_ok and ba.parsed.format_ok:
            record["y_pos"] = int(
                record["ab"]["canonical_verdict"] != record["ba"]["canonical_verdict"]
            )
        else:
            record["y_pos"] = None
        generation_cache[pair["pair_id"]] = ab

        if args.repeat_ab:
            print(f"[{index}/{len(rows)}] {pair['pair_id']}: AB2", flush=True)
            ab2 = generate_judgment(runtime, pair, "AB", args.max_new_tokens, args.context_limit)
            record["ab2"] = judgment_dict(ab2, "AB")
        records.append(record)
        append_jsonl(args.output_dir / "judgments.jsonl", record)

    extraction_errors: list[dict[str, str]] = []
    feature_rows: list[dict[str, Any]] = []
    hidden_states: list[np.ndarray] = []
    input_hidden_states: list[np.ndarray] = []
    if args.extract_features:
        for index, (pair, record) in enumerate(zip(rows, records), start=1):
            if not record["ab"]["parse"]["format_ok"]:
                continue
            print(
                f"[{index}/{len(rows)}] {pair['pair_id']}: save AB features",
                flush=True,
            )
            try:
                features = teacher_forced_extract(runtime, generation_cache[pair["pair_id"]])
            except Exception as exc:
                extraction_errors.append(
                    {"pair_id": pair["pair_id"], "error": f"{type(exc).__name__}: {exc}"}
                )
                continue

            response_a_tokens = len(
                runtime.tokenizer.encode(pair["response_a"], add_special_tokens=False)
            )
            response_b_tokens = len(
                runtime.tokenizer.encode(pair["response_b"], add_special_tokens=False)
            )
            hidden_states.append(features.hidden_states)
            input_hidden_states.append(features.input_hidden_states)
            feature_rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "y_pos": record["y_pos"],
                    "confidence": record["ab"]["parse"]["confidence"],
                    "z_a": features.z_a,
                    "z_b": features.z_b,
                    "signed_logit_margin": features.z_a - features.z_b,
                    "response_a_tokens": response_a_tokens,
                    "response_b_tokens": response_b_tokens,
                    "signed_length_difference": response_a_tokens - response_b_tokens,
                    "teacher_prefix_tokens": features.teacher_prefix_tokens,
                    "greedy_token_id": features.greedy_token_id,
                    "recorded_verdict_token_id": features.recorded_verdict_token_id,
                    "rationale_tokens": len(
                        runtime.tokenizer.encode(
                            record["ab"]["parse"]["reasoning"], add_special_tokens=False
                        )
                    ),
                    **features.uncertainty,
                }
            )

        write_jsonl(args.output_dir / "feature_metadata.jsonl", feature_rows)
        if hidden_states:
            np.savez_compressed(
                args.output_dir / "features.npz",
                pair_ids=np.asarray([row["pair_id"] for row in feature_rows]),
                layer_numbers=np.asarray(runtime.layer_numbers, dtype=np.int32),
                hidden_states=np.stack(hidden_states).astype(np.float32),
                input_hidden_states=np.stack(input_hidden_states).astype(np.float32),
            )

    orders = ["AB", "BA"] + (["AB2"] if args.repeat_ab else [])
    summary = summarize(records, orders)
    summary["feature_rows"] = len(feature_rows)
    summary["extraction_errors"] = extraction_errors
    run_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_model_revision": getattr(runtime.model.config, "_commit_hash", None),
        "resolved_tokenizer_revision": getattr(runtime.tokenizer, "_commit_hash", None)
        or runtime.tokenizer.init_kwargs.get("_commit_hash")
        or args.revision,
        "device": str(runtime.device),
        "dtype": str(runtime.dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": {
            name: version(name)
            for name in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "datasets")
        },
        "gpu_name": torch.cuda.get_device_name(runtime.device)
        if runtime.device.type == "cuda"
        else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(runtime.device).total_memory
        if runtime.device.type == "cuda"
        else None,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_hours": (time.perf_counter() - started) / 3600
        if runtime.device.type == "cuda"
        else 0.0,
        "sampling_calls_per_pair": args.samples,
        "sampling_temperature": args.temperature,
        "sampling_top_p": 1.0,
        "sampling_top_k": 0,
        "development_limit": args.limit,
        "model_parameters": sum(parameter.numel() for parameter in runtime.model.parameters()),
        "seed": args.seed,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "max_new_tokens": args.max_new_tokens,
        "context_limit": args.context_limit,
        "do_sample": False,
        "qwen_thinking": False,
        "trust_remote_code": args.trust_remote_code,
        "candidate_layers": runtime.layer_numbers,
        "feature_capture": "during original greedy AB generation",
        "verdict_token_ids": runtime.verdict_token_ids,
        "input": str(args.input),
        "summary": summary,
    }
    run_manifest["resolved_model_revision"] = (
        run_manifest["resolved_model_revision"] or args.revision
    )
    write_json(args.output_dir / "run.json", run_manifest)
    print(json.dumps(summary, indent=2))
    if extraction_errors:
        print(
            f"Excluded {len(extraction_errors)} pairs with unavailable features; see {args.output_dir / 'run.json'}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
