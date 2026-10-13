#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -eq 0 ]]; then
  gpus=(0)
else
  gpus=("$@")
fi
bash jobs/prepare_data.sh
for benchmark in judgebench llmbar mtbench; do
  python3 -m position_bias.run_suite --benchmark "$benchmark" --gpus "${gpus[@]}"
done
bash jobs/run_analysis.sh judgebench
bash jobs/run_analysis.sh llmbar
bash jobs/run_transfer.sh
bash jobs/run_controls.sh qwen3_8b
