#!/usr/bin/env bash
set -euo pipefail

python3 -m position_bias.prepare_control \
  --dataset data/context_check/analysis.jsonl \
  --common-intersection runs/analysis_judgebench/common_intersection.jsonl \
  --output-dir runs/same_order_control \
  --size 100 \
  --seed 20260828
