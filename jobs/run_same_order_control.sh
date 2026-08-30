#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 RUN_NAME" >&2
  echo "Example: $0 qwen3_1_7b" >&2
  exit 2
fi

run_name="$1"

python3 -m position_bias.run_control \
  --subset runs/same_order_control/subset.jsonl \
  --original-run "runs/${run_name}_judgebench" \
  --output-dir "runs/same_order_control/${run_name}" \
  --device cuda \
  --dtype bfloat16
