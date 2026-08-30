#!/usr/bin/env bash
set -euo pipefail

python3 -m position_bias.run_analysis \
  --judge qwen3_1_7b=runs/qwen3_1_7b_judgebench \
  --judge qwen3_4b=runs/qwen3_4b_judgebench \
  --judge qwen3_8b=runs/qwen3_8b_judgebench \
  --judge llama3_1_8b=runs/llama3_1_8b_judgebench \
  --output-dir runs/analysis_judgebench \
  --outer-folds 5 \
  --bootstrap-reps 2000
