#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
benchmark="${1:-judgebench}"
if [[ "$benchmark" != judgebench && "$benchmark" != llmbar ]]; then
  echo "Usage: bash jobs/run_analysis.sh {judgebench|llmbar}" >&2
  exit 2
fi
judges=(qwen3_1_7b qwen3_4b qwen3_8b qwen3_14b qwen3_32b llama3_1_8b granite3_1_8b mistral7b)
arguments=()
for judge in "${judges[@]}"; do
  run="runs/${judge}_${benchmark}"
  python3 -m position_bias.features --run "$run" --output "$run/embeddings.npz"
  arguments+=(--judge "$judge=$run" --embeddings "$judge=$run/embeddings.npz")
done
python3 -m position_bias.run_analysis "${arguments[@]}" --output-dir "runs/analysis_${benchmark}"
python3 -m position_bias.report --analysis "runs/analysis_${benchmark}" --output-dir "runs/report_${benchmark}"
