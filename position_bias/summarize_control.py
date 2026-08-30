from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from position_bias.io_utils import write_json


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Run must be LABEL=RUN_DIRECTORY")
    label, directory = spec.split("=", maxsplit=1)
    if not label or not directory:
        raise argparse.ArgumentTypeError("Run must be LABEL=RUN_DIRECTORY")
    return label, Path(directory)


def aggregate(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one control run is required")
    first_ids = next(iter(runs.values()))["subset_pair_ids"]
    for label, run in runs.items():
        if run["subset_pair_ids"] != first_ids:
            raise ValueError(f"{label} did not use the identical ordered subset")
        summary = run["summary"]
        if summary["pairs_completed"] != summary["pairs_requested"]:
            raise ValueError(f"{label} control run is incomplete")
    return {
        "control": "identical-order AB1 versus AB2",
        "judges": {label: run["summary"] for label, run in runs.items()},
        "subset_n": len(first_ids),
        "subset_pair_ids": first_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed repeated-AB control runs.")
    parser.add_argument(
        "--run", action="append", required=True, type=parse_run_spec, metavar="LABEL=DIR"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_specs = dict(args.run)
    if len(run_specs) != len(args.run):
        raise ValueError("Control labels must be unique")
    runs = {
        label: json.loads((directory / "run.json").read_text(encoding="utf-8"))
        for label, directory in run_specs.items()
    }
    results = aggregate(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "results.json", results)
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "judge",
                "subset_n",
                "ab2_format_ok",
                "same_order_disagreements",
                "same_order_rate",
                "swap_disagreements",
                "swap_rate_on_subset",
            ]
        )
        for label, summary in results["judges"].items():
            writer.writerow(
                [
                    label,
                    results["subset_n"],
                    summary["ab2_format_ok"],
                    summary["same_order_disagreements"],
                    summary["same_order_rate"],
                    summary["swap_disagreements"],
                    summary["swap_rate_on_subset"],
                ]
            )
    print(f"Same-order control: {results['subset_n']} shared pairs")
    for label, summary in results["judges"].items():
        print(
            f"{label}: AB1/AB2 {summary['same_order_disagreements']}/"
            f"{summary['same_order_usable']} vs AB/BA {summary['swap_disagreements']}/"
            f"{summary['swap_usable']}"
        )


if __name__ == "__main__":
    main()
