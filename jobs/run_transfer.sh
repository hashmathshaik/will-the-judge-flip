#!/usr/bin/env bash
set -euo pipefail

python3 -m position_bias.run_transfer \
  --id-judge qwen3_1_7b=runs/qwen3_1_7b_judgebench \
  --id-judge qwen3_4b=runs/qwen3_4b_judgebench \
  --id-judge qwen3_8b=runs/qwen3_8b_judgebench \
  --id-judge llama3_1_8b=runs/llama3_1_8b_judgebench \
  --ood-judge qwen3_1_7b=runs/qwen3_1_7b_mtbench \
  --ood-judge qwen3_4b=runs/qwen3_4b_mtbench \
  --ood-judge qwen3_8b=runs/qwen3_8b_mtbench \
  --ood-judge llama3_1_8b=runs/llama3_1_8b_mtbench \
  --output-dir runs/judgebench_to_mtbench \
  --selection-folds 5 \
  --bootstrap-reps 2000
