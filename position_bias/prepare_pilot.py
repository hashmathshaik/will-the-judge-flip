from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from position_bias.io_utils import read_jsonl, write_json, write_jsonl
from position_bias.protocol import judge_messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="qwen3_1_7b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--judgebench-pairs", type=int, default=24)
    parser.add_argument("--llmbar-pairs", type=int, default=20)
    parser.add_argument("--mtbench-pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        min(args.max_prompt_tokens, args.judgebench_pairs, args.llmbar_pairs, args.mtbench_pairs)
        < 1
    ):
        raise ValueError("Pilot sizes and prompt limit must be positive")
    model = json.loads(Path("configs/models.json").read_text())[args.judge]
    tokenizer = AutoTokenizer.from_pretrained(
        model["model_id"], revision=model["revision"], cache_dir="data/hf_cache"
    )
    reports = {}
    for benchmark, count in (
        ("judgebench", args.judgebench_pairs),
        ("llmbar", args.llmbar_pairs),
        ("mtbench", args.mtbench_pairs),
    ):
        rows = read_jsonl(Path("data") / (benchmark + ".jsonl"))
        eligible = []
        for row in rows:
            lengths = [
                len(
                    tokenizer.apply_chat_template(
                        judge_messages(row, order),
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                for order in ("AB", "BA")
            ]
            if max(lengths) <= args.max_prompt_tokens:
                eligible.append((row, max(lengths)))
        buckets = {}
        for row, length in eligible:
            key = row["source"] if benchmark == "llmbar" else row["coarse_domain"]
            buckets.setdefault(key, []).append((row, length))
        for bucket in buckets.values():
            bucket.sort(
                key=lambda item: hashlib.sha256(
                    f"{args.seed}:{item[0]['pair_id']}".encode()
                ).hexdigest()
            )
        chosen, groups = [], set()
        while len(chosen) < count:
            changed = False
            for key in sorted(buckets):
                while buckets[key] and buckets[key][0][0]["group_id"] in groups:
                    buckets[key].pop(0)
                if buckets[key] and len(chosen) < count:
                    item = buckets[key].pop(0)
                    chosen.append(item)
                    groups.add(item[0]["group_id"])
                    changed = True
            if not changed:
                break
        if len(chosen) != count:
            raise ValueError(
                f"{benchmark}: only {len(chosen)} distinct eligible questions for {count} requested pairs"
            )
        write_jsonl(args.output_dir / "inputs" / (benchmark + ".jsonl"), [row for row, _ in chosen])
        reports[benchmark] = {
            "available": len(rows),
            "eligible": len(eligible),
            "selected": len(chosen),
            "question_groups": len(groups),
            "max_prompt_tokens": max(length for _, length in chosen),
            "pair_ids": [row["pair_id"] for row, _ in chosen],
        }
    write_json(
        args.output_dir / "selection.json",
        {
            "judge": args.judge,
            "model": model,
            "seed": args.seed,
            "selection": "label-independent hash order, round-robin domains/subsets, unique question groups",
            "max_prompt_tokens": args.max_prompt_tokens,
            "benchmarks": reports,
            "purpose": "label-independent benchmark pilot",
        },
    )


if __name__ == "__main__":
    main()
