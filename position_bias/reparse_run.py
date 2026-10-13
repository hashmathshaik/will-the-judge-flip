from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from position_bias.io_utils import read_jsonl, write_json, write_jsonl
from position_bias.modeling import GeneratedJudgment, load_runtime, teacher_forced_extract
from position_bias.protocol import (
    PROTOCOL_VERSION,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    canonicalize,
    parse_completion,
)
from position_bias.run_judge import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    manifest_path = args.run / "run.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["system_prompt"] != SYSTEM_PROMPT or manifest["user_template"] != USER_TEMPLATE:
        raise ValueError("Saved judgments used a different prompt")
    if args.output_dir.exists():
        raise FileExistsError("Reparsing requires a fresh output directory")
    inputs = {r["pair_id"]: r for r in read_jsonl(args.input)}
    records = read_jsonl(args.run / "judgments.jsonl")
    old_meta = {r["pair_id"]: r for r in read_jsonl(args.run / "feature_metadata.jsonl")}
    old_hidden, old_input = {}, {}
    if (args.run / "features.npz").exists():
        with np.load(args.run / "features.npz", allow_pickle=False) as data:
            old_hidden = {str(p): data["hidden_states"][i] for i, p in enumerate(data["pair_ids"])}
            old_input = {
                str(p): data["input_hidden_states"][i] for i, p in enumerate(data["pair_ids"])
            }
    metadata, hidden, input_hidden, errors = [], [], [], []
    runtime, recovered = None, 0
    for record in records:
        pair_id = record["pair_id"]
        if record.get("exclusion"):
            continue
        for order in ("ab", "ba", "ab2"):
            if order not in record:
                continue
            judgment = record[order]
            parsed = parse_completion(judgment["completion"])
            judgment["previous_parse"] = judgment["parse"]
            judgment["parse"] = parsed.to_dict()
            judgment["canonical_verdict"] = (
                canonicalize(parsed.slot, "BA" if order == "ba" else "AB")
                if parsed.format_ok
                else None
            )
        for judgment in record.get("samples", []):
            parsed = parse_completion(judgment["completion"])
            judgment["parse"] = parsed.to_dict()
            judgment["canonical_verdict"] = (
                canonicalize(parsed.slot, "AB") if parsed.format_ok else None
            )
        record["y_pos"] = (
            int(record["ab"]["canonical_verdict"] != record["ba"]["canonical_verdict"])
            if record["ab"]["parse"]["format_ok"] and record["ba"]["parse"]["format_ok"]
            else None
        )
        if not record["ab"]["parse"]["format_ok"]:
            continue
        parsed = parse_completion(record["ab"]["completion"])
        previous = record["ab"]["previous_parse"]
        if pair_id in old_hidden and previous.get("rationale_end") == parsed.rationale_end:
            meta = {**old_meta[pair_id], "confidence": parsed.confidence, "y_pos": record["y_pos"]}
            h, h_input = old_hidden[pair_id], old_input[pair_id]
        else:
            if runtime is None:
                runtime = load_runtime(
                    manifest["model"],
                    manifest["requested_revision"],
                    Path("data/hf_cache"),
                    args.device,
                    manifest["dtype"].removeprefix("torch."),
                    True,
                    8,
                )
            raw = record["ab"]
            judgment = GeneratedJudgment(
                raw["prompt_ids"],
                raw["completion_ids"],
                raw["completion"],
                parsed,
                raw["stopped_on_eos"],
            )
            try:
                features = teacher_forced_extract(runtime, judgment)
            except (ValueError, RuntimeError) as exc:
                errors.append({"pair_id": pair_id, "error": str(exc)})
                continue
            pair = inputs[pair_id]
            a_len = len(runtime.tokenizer.encode(pair["response_a"], add_special_tokens=False))
            b_len = len(runtime.tokenizer.encode(pair["response_b"], add_special_tokens=False))
            meta = {
                "pair_id": pair_id,
                "confidence": parsed.confidence,
                "y_pos": record["y_pos"],
                "z_a": features.z_a,
                "z_b": features.z_b,
                "signed_logit_margin": features.z_a - features.z_b,
                "response_a_tokens": a_len,
                "response_b_tokens": b_len,
                "signed_length_difference": a_len - b_len,
                "rationale_tokens": len(
                    runtime.tokenizer.encode(parsed.reasoning, add_special_tokens=False)
                ),
                "teacher_prefix_tokens": features.teacher_prefix_tokens,
                "greedy_token_id": features.greedy_token_id,
                "recorded_verdict_token_id": features.recorded_verdict_token_id,
                **features.uncertainty,
            }
            h, h_input = features.hidden_states, features.input_hidden_states
            recovered += 1
        metadata.append(meta)
        hidden.append(h)
        input_hidden.append(h_input)
    write_jsonl(args.output_dir / "judgments.jsonl", records)
    write_jsonl(args.output_dir / "feature_metadata.jsonl", metadata)
    if hidden:
        np.savez_compressed(
            args.output_dir / "features.npz",
            pair_ids=np.asarray([r["pair_id"] for r in metadata]),
            layer_numbers=np.asarray(manifest["candidate_layers"]),
            hidden_states=np.stack(hidden),
            input_hidden_states=np.stack(input_hidden),
        )
    summary = summarize(
        records, ["AB", "BA"] + (["AB2"] if any("ab2" in r for r in records) else [])
    )
    summary["feature_rows"] = len(metadata)
    summary["extraction_errors"] = errors
    write_json(
        args.output_dir / "run.json",
        {
            **manifest,
            "protocol_version": PROTOCOL_VERSION,
            "feature_capture": "greedy AB trace: original capture or verified cached replay",
            "summary": summary,
            "reparsing": {
                "original_run": str(args.run),
                "original_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "previous_protocol": manifest["protocol_version"],
                "recovered_by_replay": recovered,
                "new_judgment_calls": 0,
            },
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
