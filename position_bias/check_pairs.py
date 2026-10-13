from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from position_bias.artifacts import load_model
from position_bias.audit import rank_pairs
from position_bias.io_utils import read_jsonl, write_jsonl
from position_bias.modeling import generate_judgment, load_runtime, teacher_forced_extract
from position_bias.protocol import PROTOCOL_VERSION, canonicalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--source-analysis",
        type=Path,
        required=True,
        help="One judge's frozen source-analysis directory",
    )
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 0 <= args.budget <= 1:
        raise ValueError("Budget must lie in [0,1]")
    frozen = json.loads((args.source_analysis / "frozen.json").read_text())
    protocol = frozen["protocol"]
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Frozen predictor uses a different prompt protocol")
    rows = read_jsonl(args.input)
    if not rows or len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("Input must be nonempty with unique pair IDs")
    runtime = load_runtime(
        protocol["model"],
        protocol["requested_revision"],
        Path("data/hf_cache"),
        args.device,
        protocol["dtype"].removeprefix("torch."),
        False,
        8,
    )
    entry = frozen["predictors"]["probe"]
    model = load_model(args.source_analysis / entry["model_file"])
    scored, uncertain = [], []
    for row in rows:
        judgment = generate_judgment(
            runtime,
            row,
            "AB",
            protocol["max_new_tokens"],
            protocol["context_limit"],
            capture_features=True,
        )
        if not judgment.parsed.format_ok:
            uncertain.append(
                {
                    "pair_id": row["pair_id"],
                    "status": "escalate: malformed AB",
                    "ab": judgment.completion,
                }
            )
            continue
        if judgment.feature_error is not None:
            uncertain.append(
                {
                    "pair_id": row["pair_id"],
                    "status": "escalate: unavailable features",
                    "ab": judgment.completion,
                    "error": judgment.feature_error,
                }
            )
            continue
        features = teacher_forced_extract(runtime, judgment)
        risk = float(
            model.predict(features.hidden_states[int(entry["selected"]["layer_index"])][None, :])[0]
        )
        scored.append((row, judgment, risk))
    ranked = (
        rank_pairs(np.asarray([s[2] for s in scored]), [s[0]["pair_id"] for s in scored])
        if scored
        else []
    )
    selected = set(ranked[: round(args.budget * len(scored))])
    outputs = []
    for i, (row, ab, risk) in enumerate(scored):
        item = {
            "pair_id": row["pair_id"],
            "risk": risk,
            "ab": ab.completion,
            "checked": i in selected,
        }
        if i in selected:
            ba = generate_judgment(
                runtime, row, "BA", protocol["max_new_tokens"], protocol["context_limit"]
            )
            item["ba"] = ba.completion
            item["reversal"] = (
                int(canonicalize(ab.parsed.slot, "AB") != canonicalize(ba.parsed.slot, "BA"))
                if ba.parsed.format_ok
                else None
            )
            item["status"] = (
                "escalate: malformed BA"
                if not ba.parsed.format_ok
                else "disagreement"
                if item["reversal"]
                else "stable under reorder"
            )
        else:
            item["status"] = "unchecked"
        outputs.append(item)
    write_jsonl(args.output, outputs + uncertain)


if __name__ == "__main__":
    main()
