#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$#" -gt 6 ]]; then
  echo "Usage: $0 {judgebench|mtbench} MODEL_ID EXACT_REVISION RUN_NAME [CONTEXT_LIMIT] [--trust-remote-code]" >&2
  exit 2
fi

benchmark="$1"
model_id="$2"
model_revision="$3"
run_name="$4"
context_limit="${5:-32768}"
remote_code_flag="${6:-}"

extra_args=()
if [[ -n "$remote_code_flag" ]]; then
  if [[ "$remote_code_flag" != "--trust-remote-code" ]]; then
    echo "Sixth argument must be --trust-remote-code" >&2
    exit 2
  fi
  extra_args+=("$remote_code_flag")
fi

case "$benchmark" in
  judgebench)
    input_path="data/context_check/analysis.jsonl"
    output_path="runs/${run_name}_judgebench"
    ;;
  mtbench)
    input_path="data/mtbench_context/analysis.jsonl"
    output_path="runs/${run_name}_mtbench"
    ;;
  *)
    echo "Benchmark must be judgebench or mtbench" >&2
    exit 2
    ;;
esac

python3 -m position_bias.run_judge \
  --input "$input_path" \
  --output-dir "$output_path" \
  --model "$model_id" \
  --revision "$model_revision" \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 256 \
  --context-limit "$context_limit" \
  --extract-features \
  "${extra_args[@]}"
