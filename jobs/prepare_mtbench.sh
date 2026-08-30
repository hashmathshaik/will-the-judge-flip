#!/usr/bin/env bash
set -euo pipefail

python3 -m position_bias.prepare_mtbench

python3 -m position_bias.check_context \
  --input data/mtbench.jsonl \
  --output-dir data/mtbench_context \
  --tokenizer qwen3=Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218 \
  --tokenizer llama3_1=meta-llama/Meta-Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659 \
  --context-limit 32768 \
  --primary-allowance 256 \
  --minimum-retention 0.60
