from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from position_bias.artifacts import load_model
from position_bias.audit import rank_pairs
from position_bias.io_utils import read_jsonl, write_json
from position_bias.modeling import generate_judgment, load_runtime, teacher_forced_extract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--scored-run",
        type=Path,
        help="Restrict timing to the retained cohort of a completed target run",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.budget <= 1 or args.warmups < 0 or args.repeats < 1:
        raise ValueError("Invalid budget, warmup count, or repeats")
    frozen = json.loads((args.source_analysis / "frozen.json").read_text())
    config, entry = frozen["protocol"], frozen["predictors"]["probe"]
    rows = read_jsonl(args.input)
    if args.scored_run:
        from position_bias.artifacts import check_frozen_protocol, load_judge

        scored = load_judge(args.scored_run, include_hidden=False)
        check_frozen_protocol(config, scored["manifest"])
        rows = [row for row in rows if row["pair_id"] in scored["usable"]]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Runtime input is empty")
    runtime = load_runtime(
        config["model"],
        config["requested_revision"],
        Path("data/hf_cache"),
        args.device,
        config["dtype"].removeprefix("torch."),
        False,
        8,
    )
    model = load_model(args.source_analysis / entry["model_file"])

    def generation(row, order, capture=False):
        return generate_judgment(
            runtime,
            row,
            order,
            config["max_new_tokens"],
            config["context_limit"],
            capture_features=capture,
        )

    def sync():
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)

    for i in range(args.warmups):
        generation(rows[i % len(rows)], "AB", True)
    repeats = []
    for repeat in range(args.repeats):
        durations = {}

        for strategy in (
            ("exhaustive", "selective") if repeat % 2 == 0 else ("selective", "exhaustive")
        ):
            sync()
            start = time.perf_counter()
            if strategy == "exhaustive":
                for row in rows:
                    generation(row, "AB")
                    generation(row, "BA")
            else:
                risks = []
                for row in rows:
                    ab = generation(row, "AB", True)
                    if not ab.parsed.format_ok:
                        raise ValueError(
                            "Runtime measurement requires a fully parse-valid scored cohort"
                        )
                    features = teacher_forced_extract(runtime, ab)
                    risks.append(
                        float(
                            model.predict(
                                features.hidden_states[int(entry["selected"]["layer_index"])][
                                    None, :
                                ]
                            )[0]
                        )
                    )
                ranked = rank_pairs(np.asarray(risks), [r["pair_id"] for r in rows])
                for i in ranked[: round(args.budget * len(rows))]:
                    generation(rows[i], "BA")
            sync()
            durations[strategy + "_seconds"] = time.perf_counter() - start
        repeats.append(durations)
    full = np.asarray([r["exhaustive_seconds"] for r in repeats])
    selective = np.asarray([r["selective_seconds"] for r in repeats])
    write_json(
        args.output,
        {
            "n": len(rows),
            "budget": args.budget,
            "batch_size": 1,
            "warmups": args.warmups,
            "repeats": repeats,
            "aggregation": "mean",
            "exhaustive_mean_seconds": float(full.mean()),
            "exhaustive_std_seconds": float(full.std()),
            "selective_mean_seconds": float(selective.mean()),
            "selective_std_seconds": float(selective.std()),
            "runtime_reduction": float(1 - selective.mean() / full.mean()),
            "includes": "generation-time activation capture, preprocessing, readout, ranking, selected BA calls",
            "excludes": "model loading, training, label acquisition, warmups",
            "device": str(runtime.device),
            "gpu_name": torch.cuda.get_device_name(runtime.device)
            if runtime.device.type == "cuda"
            else None,
            "dtype": str(runtime.dtype),
        },
    )


if __name__ == "__main__":
    main()
