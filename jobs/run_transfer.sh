#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
judges=(qwen3_1_7b qwen3_4b qwen3_8b qwen3_14b qwen3_32b llama3_1_8b granite3_1_8b mistral7b)
arguments=()
for judge in "${judges[@]}"; do
  arguments+=(--judge "$judge=runs/${judge}_mtbench")
done
python3 -m position_bias.run_transfer "${arguments[@]}" --source-analysis runs/analysis_judgebench --output-dir runs/transfer_mtbench
python3 -m position_bias.run_transfer "${arguments[@]}" --source-analysis runs/analysis_judgebench --output-dir runs/transfer_mtbench_common --population common
python3 -m position_bias.report --analysis runs/transfer_mtbench --output-dir runs/report_mtbench
