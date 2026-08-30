from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import transformers

from position_bias.io_utils import read_jsonl, write_json, write_jsonl
from position_bias.modeling import generate_judgment, load_runtime, render_prompt_ids
from position_bias.run_judge import judgment_dict


def _records_by_id(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id in records:
            raise ValueError(f"Duplicate pair_id {pair_id!r} in {source}")
        records[pair_id] = row
    return records


def summarize(records: list[dict[str, Any]], requested: int) -> dict[str, Any]:
    ab2_valid = [record for record in records if record["ab2"]["parse"]["format_ok"]]
    same_order = [record for record in ab2_valid if record["ab1"]["format_ok"]]
    swap = [
        record for record in records if record["ab1"]["format_ok"] and record["ba"]["format_ok"]
    ]
    same_disagreements = sum(record["same_order_disagreement"] for record in same_order)
    swap_disagreements = sum(record["position_flip"] for record in swap)
    return {
        "pairs_requested": requested,
        "pairs_completed": len(records),
        "ab2_format_ok": len(ab2_valid),
        "ab2_format_rate": len(ab2_valid) / len(records) if records else None,
        "same_order_usable": len(same_order),
        "same_order_disagreements": same_disagreements,
        "same_order_rate": same_disagreements / len(same_order) if same_order else None,
        "swap_usable": len(swap),
        "swap_disagreements": swap_disagreements,
        "swap_rate_on_subset": swap_disagreements / len(swap) if swap else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat AB judgments and compare them with saved AB and BA verdicts."
    )
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("auto", "float32", "bfloat16", "float16")
    )
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = read_jsonl(args.subset)
    if not subset:
        raise ValueError("The control subset is empty")
    subset_by_id = _records_by_id(subset, "control subset")

    original_manifest = json.loads((args.original_run / "run.json").read_text(encoding="utf-8"))
    if original_manifest.get("do_sample") is not False:
        raise ValueError("The original run was not deterministic greedy decoding")
    if original_manifest.get("qwen_thinking") is not False:
        raise ValueError("The original run did not use the frozen non-thinking protocol")
    original_rows = read_jsonl(args.original_run / "judgments.jsonl")
    original_by_id = _records_by_id(original_rows, "original judgments")
    missing = sorted(set(subset_by_id) - set(original_by_id))
    if missing:
        raise ValueError(f"{len(missing)} control pairs are absent from the original run")
    for pair_id in subset_by_id:
        original = original_by_id[pair_id]
        if not original["ab"]["parse"]["format_ok"] or not original["ba"]["parse"]["format_ok"]:
            raise ValueError(f"Control pair {pair_id} is not AB/BA parse-valid")

    model_id = str(original_manifest["model"])
    revision = str(original_manifest["requested_revision"])
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("The original run does not contain an exact model revision")
    max_new_tokens = int(original_manifest["max_new_tokens"])
    context_limit = int(original_manifest["context_limit"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = args.output_dir / "judgments.jsonl"
    existing = read_jsonl(judgments_path) if judgments_path.exists() else []
    expected_ids = list(subset_by_id)
    existing_ids = [str(record["pair_id"]) for record in existing]
    if existing_ids != expected_ids[: len(existing_ids)]:
        raise ValueError("Existing control records are not a prefix of the frozen subset")
    if len(existing) > len(subset):
        raise ValueError("Existing control output is longer than the frozen subset")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    runtime = None
    if len(existing) < len(subset):
        print(f"Loading {model_id}@{revision} for {len(subset) - len(existing)} AB2 judgments...")
        runtime = load_runtime(
            model_or_path=model_id,
            revision=revision,
            cache_dir=args.cache_dir,
            requested_device=args.device,
            requested_dtype=args.dtype,
            local_files_only=args.local_files_only,
            cpu_threads=args.cpu_threads,
            trust_remote_code=bool(original_manifest.get("trust_remote_code", False)),
        )
        if str(runtime.dtype) != str(original_manifest["dtype"]):
            raise ValueError(
                f"Control dtype {runtime.dtype} differs from original {original_manifest['dtype']}"
            )

    records = list(existing)
    for index in range(len(existing), len(subset)):
        pair = subset[index]
        pair_id = str(pair["pair_id"])
        original = original_by_id[pair_id]
        assert runtime is not None
        prompt_ids = render_prompt_ids(runtime, pair, "AB")
        if prompt_ids != original["ab"]["prompt_ids"]:
            raise ValueError(f"Frozen AB prompt changed for {pair_id}")

        print(f"[{index + 1}/{len(subset)}] {pair_id}: AB2", flush=True)
        ab2_generated = generate_judgment(runtime, pair, "AB", max_new_tokens, context_limit)
        ab2 = judgment_dict(ab2_generated, "AB")
        ab1 = {
            "canonical_verdict": original["ab"]["canonical_verdict"],
            "format_ok": original["ab"]["parse"]["format_ok"],
            "slot": original["ab"]["parse"]["slot"],
        }
        ba = {
            "canonical_verdict": original["ba"]["canonical_verdict"],
            "format_ok": original["ba"]["parse"]["format_ok"],
            "slot": original["ba"]["parse"]["slot"],
        }
        record = {
            "pair_id": pair_id,
            "group_id": pair["group_id"],
            "source": pair["source"],
            "ab1": ab1,
            "ba": ba,
            "ab2": ab2,
            "position_flip": int(original["y_pos"]),
            "same_order_disagreement": (
                int(ab1["canonical_verdict"] != ab2["canonical_verdict"])
                if ab2["parse"]["format_ok"]
                else None
            ),
        }
        records.append(record)
        write_jsonl(judgments_path, records)

    summary = summarize(records, requested=len(subset))
    write_json(
        args.output_dir / "run.json",
        {
            "control": "identical-order AB1 versus AB2",
            "device": str(runtime.device) if runtime is not None else args.device,
            "do_sample": False,
            "dtype": str(runtime.dtype) if runtime is not None else args.dtype,
            "max_new_tokens": max_new_tokens,
            "model": model_id,
            "original_run": str(args.original_run),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "qwen_thinking": False,
            "requested_revision": revision,
            "subset": str(args.subset),
            "subset_pair_ids": expected_ids,
            "summary": summary,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
