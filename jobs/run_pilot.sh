#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pilot_dir="${1:?Supply a fresh pilot output directory}"
device="${2:-cuda}"
judge="${3:-qwen3_1_7b}"
read -r model revision < <(python3 -c 'import json,sys; m=json.load(open("configs/models.json"))[sys.argv[1]]; print(m["model_id"], m["revision"])' "$judge")
dtype=bfloat16
if [[ "$device" == cpu ]]; then
  dtype=float32
fi
python3 -m position_bias.prepare_pilot --judge "$judge" --output-dir "$pilot_dir"
for benchmark in judgebench llmbar mtbench; do
  samples=5
  if [[ "$benchmark" == mtbench ]]; then samples=0; fi
  python3 -u -m position_bias.run_judge \
    --input "$pilot_dir/inputs/$benchmark.jsonl" \
    --output-dir "$pilot_dir/${judge}_$benchmark" \
    --model "$model" --revision "$revision" \
    --device "$device" --dtype "$dtype" --samples "$samples" --extract-features
done
for benchmark in judgebench llmbar; do
  run="$pilot_dir/${judge}_$benchmark"
  python3 -m position_bias.features --run "$run" --output "$run/embeddings.npz"
  python3 -m position_bias.run_analysis \
    --judge "$judge=$run" --embeddings "$judge=$run/embeddings.npz" \
    --output-dir "$pilot_dir/analysis_$benchmark"
  python3 -m position_bias.report --analysis "$pilot_dir/analysis_$benchmark" --output-dir "$pilot_dir/report_$benchmark"
done
python3 -m position_bias.run_transfer \
  --judge "$judge=$pilot_dir/${judge}_mtbench" \
  --source-analysis "$pilot_dir/analysis_judgebench" --output-dir "$pilot_dir/transfer_mtbench"
python3 -m position_bias.report --analysis "$pilot_dir/transfer_mtbench" --output-dir "$pilot_dir/report_mtbench"
python3 -m position_bias.summarize_pilot --pilot-dir "$pilot_dir"
