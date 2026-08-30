#!/usr/bin/env bash
set -euo pipefail

python3 -m position_bias.summarize_control \
  --run qwen3_1_7b=runs/same_order_control/qwen3_1_7b \
  --run qwen3_4b=runs/same_order_control/qwen3_4b \
  --run qwen3_8b=runs/same_order_control/qwen3_8b \
  --run llama3_1_8b=runs/same_order_control/llama3_1_8b \
  --output-dir runs/same_order_control/aggregate
