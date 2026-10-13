#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m position_bias.prepare_judgebench
python3 -m position_bias.prepare_llmbar
python3 -m position_bias.prepare_mtbench
