#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
judge="${1:-qwen3_8b}"
python3 -m position_bias.controls --run "runs/${judge}_judgebench" --analysis "runs/analysis_judgebench/$judge" --embeddings "runs/${judge}_judgebench/embeddings.npz" --output-dir "runs/controls_$judge"
