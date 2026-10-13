from __future__ import annotations

import argparse
import json
from pathlib import Path

from position_bias.io_utils import read_jsonl, write_json, write_jsonl
from position_bias.modeling import generate_judgment, load_runtime
from position_bias.protocol import PROTOCOL_VERSION, canonicalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--paraphrases",
        type=Path,
        help="JSONL with pair_id and paraphrased question",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = json.loads((args.original_run / "run.json").read_text())
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Original run uses a different protocol")
    originals = {r["pair_id"]: r for r in read_jsonl(args.original_run / "judgments.jsonl")}
    paraphrases = (
        {r["pair_id"]: r["question"] for r in read_jsonl(args.paraphrases)}
        if args.paraphrases
        else {}
    )
    runtime = load_runtime(
        manifest["model"],
        manifest["requested_revision"],
        Path("data/hf_cache"),
        args.device,
        manifest["dtype"].removeprefix("torch."),
        False,
        8,
    )
    records = []
    for row in read_jsonl(args.input):
        original = originals.get(row["pair_id"])
        if not original or not original["ab"]["parse"]["format_ok"]:
            continue
        edits = {
            "whitespace": {
                **row,
                "question": row["question"] + "\n",
                "response_a": row["response_a"] + "\n",
                "response_b": row["response_b"] + "\n",
            }
        }
        if row["pair_id"] in paraphrases:
            edits["paraphrase"] = {**row, "question": paraphrases[row["pair_id"]]}
        for kind, edited in edits.items():
            ab = generate_judgment(
                runtime, edited, "AB", manifest["max_new_tokens"], manifest["context_limit"]
            )
            verdict = canonicalize(ab.parsed.slot, "AB") if ab.parsed.format_ok else None
            records.append(
                {
                    "pair_id": row["pair_id"],
                    "perturbation": kind,
                    "completion": ab.completion,
                    "valid": ab.parsed.format_ok,
                    "changed": int(verdict != original["ab"]["canonical_verdict"])
                    if verdict
                    else None,
                }
            )
    write_jsonl(args.output_dir / "judgments.jsonl", records)
    report = {}
    for kind in ("whitespace", "paraphrase"):
        valid = [r for r in records if r["perturbation"] == kind and r["valid"]]
        report[kind] = {
            "n": len(valid),
            "changed": sum(r["changed"] for r in valid),
            "rate": sum(r["changed"] for r in valid) / len(valid) if valid else None,
        }
    write_json(
        args.output_dir / "results.json",
        {
            "results": report,
            "whitespace_rule": "append one newline to request and both responses",
            "paraphrases_input": str(args.paraphrases) if args.paraphrases else None,
        },
    )


if __name__ == "__main__":
    main()
