from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PAPER_LABELS = {
    "qwen3_1_7b": "Qwen3-1.7B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "llama3_1_8b": "Llama-3.1-8B",
}


def trim(value: float | None, places: int = 3) -> str:
    if value is None:
        return "NE"
    text = f"{value:.{places}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def signed(value: float | None, places: int = 3) -> str:
    if value is None:
        return "NE"
    return ("+" if value >= 0 else "") + trim(value, places)


def interval(entry: dict[str, Any], metric: str, places: int = 3) -> str:
    bounds = entry["bootstrap_95_ci"][metric]
    return f"[{trim(bounds['lower'], places)},{trim(bounds['upper'], places)}]"


def direct_interval(entry: dict[str, Any], places: int = 3) -> str:
    bounds = entry["bootstrap_95_ci"]
    return f"[{trim(bounds['lower'], places)},{trim(bounds['upper'], places)}]"


def paper_label(label: str) -> str:
    return PAPER_LABELS.get(label, label)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit LaTeX table rows from saved experiment results."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--table",
        required=True,
        choices=(
            "main",
            "baselines",
            "appendix_metrics",
            "paired",
            "stratified",
        ),
    )
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    stored = results["judges"]
    ranked = [label for label in PAPER_LABELS if label in stored]
    ranked += [label for label in stored if label not in PAPER_LABELS]
    judges = {label: stored[label] for label in ranked}

    if args.table == "main":
        for label, judge in judges.items():
            entry = judge["predictors"]["hidden_state"]
            metrics = entry["metrics"]
            print(
                f"{paper_label(label):12s} & {judge['positive_count']}/{judge['n']} "
                f"& {trim(judge['pfr'])} "
                f"& {trim(metrics['auroc'])} {interval(entry, 'auroc')} "
                f"& {trim(metrics['auprc'])} {interval(entry, 'auprc')} "
                f"& {trim(metrics['brier'])} {interval(entry, 'brier')} \\\\"
            )

    elif args.table == "baselines":
        order = [
            "hidden_state",
            "combined_output_slot",
            "combined_output",
            "logit_margin",
            "verdict_slot",
            "confidence",
            "length_difference",
        ]
        for label, judge in judges.items():
            cells = []
            for name in order:
                cell = trim(judge["predictors"][name]["metrics"]["auroc"])
                cells.append(f"\\textbf{{{cell}}}" if name == "hidden_state" else cell)
            print(f"{paper_label(label):12s} & " + " & ".join(cells) + " \\\\")
        for label, judge in judges.items():
            cells = []
            for name in order:
                if name == "verdict_slot":
                    cells.append("")
                    continue
                cell = trim(judge["slot_stratified"][name]["auroc"])
                cells.append(f"\\textbf{{{cell}}}" if name == "hidden_state" else cell)
            print(f"{paper_label(label):12s} & " + " & ".join(cells) + " \\\\")

    elif args.table == "appendix_metrics":
        order = [
            "hidden_state",
            "combined_output_slot",
            "combined_output",
            "logit_margin",
            "verdict_slot",
            "confidence",
            "length_difference",
            "no_signal",
        ]
        for label, judge in judges.items():
            cells = []
            for name in order:
                metrics = judge["predictors"][name]["metrics"]
                cells.append(f"{trim(metrics['auprc'])}/{trim(metrics['brier'])}")
            print(f"{paper_label(label):12s} & " + " & ".join(cells) + " \\\\")

    elif args.table == "paired":
        order = [
            "combined_output_slot",
            "combined_output",
            "logit_margin",
            "verdict_slot",
        ]
        for label, judge in judges.items():
            cells = []
            for name in order:
                bounds = judge["paired_probe_minus_baseline"][name]["auroc"]
                midpoint = (
                    judge["predictors"]["hidden_state"]["metrics"]["auroc"]
                    - judge["predictors"][name]["metrics"]["auroc"]
                )
                cells.append(
                    f"{signed(midpoint)} [{trim(bounds['lower'])},{trim(bounds['upper'])}]"
                )
            print(f"{paper_label(label):12s} & " + " & ".join(cells) + " \\\\")
        for label, judge in judges.items():
            cells = []
            for name in order:
                if name == "verdict_slot":
                    cells.append("")
                    continue
                bounds = judge["paired_probe_minus_baseline_slot_stratified"][name]
                midpoint = (
                    judge["slot_stratified"]["hidden_state"]["auroc"]
                    - judge["slot_stratified"][name]["auroc"]
                )
                cells.append(
                    f"{signed(midpoint)} [{trim(bounds['lower'])},{trim(bounds['upper'])}]"
                )
            print(f"{paper_label(label):12s} & " + " & ".join(cells) + " \\\\")

    elif args.table == "stratified":
        order = ["hidden_state", "combined_output_slot", "combined_output", "logit_margin"]
        for label, judge in judges.items():
            cells = []
            for name in order:
                entry = judge["slot_stratified"][name]
                cell = f"{trim(entry['auroc'])} {direct_interval(entry)}"
                cells.append(f"\\textbf{{{cell}}}" if name == "hidden_state" else cell)
            print(f"{paper_label(label):12s} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
