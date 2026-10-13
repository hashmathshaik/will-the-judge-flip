from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from position_bias.io_utils import read_jsonl, write_json, write_jsonl
from position_bias.prepare_judgebench import exact_pair_key, normalize_whitespace

DEFAULT_REVISION = "900616bff90b6c6c8e1681f7d079250637c55992"
SUBSETS = (
    "Natural",
    "Adversarial/Neighbor",
    "Adversarial/GPTInst",
    "Adversarial/GPTOut",
    "Adversarial/Manual",
)


def normalize_row(raw: dict, subset: str, index: int) -> dict:
    question = str(raw["input"])
    digest = hashlib.sha256(normalize_whitespace(question).encode()).hexdigest()
    return {
        "pair_id": f"llmbar::{subset}::{index}",
        "group_id": "llmbar::q" + digest,
        "question": question,
        "response_a": str(raw["output_1"]),
        "response_b": str(raw["output_2"]),
        "source": "llmbar/" + subset,
        "coarse_domain": "instruction-following",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--input-dir", type=Path, help="Offline directory containing the five subset JSON files"
    )
    parser.add_argument("--output", type=Path, default=Path("data/llmbar.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/llmbar_prepare_report.json"))
    parser.add_argument(
        "--exclude-overlap-with",
        type=Path,
        help="Optional question/pair overlap policy; not assumed from the manuscript",
    )
    args = parser.parse_args()
    if len(args.revision) != 40:
        raise ValueError("LLMBar revision must be a pinned Git commit")
    overlaps = read_jsonl(args.exclude_overlap_with) if args.exclude_overlap_with else []
    overlap_questions = {normalize_whitespace(r["question"]) for r in overlaps}
    overlap_pairs = {exact_pair_key(r) for r in overlaps}
    selected, raw_counts, excluded = [], {}, []
    seen = set()
    for subset in SUBSETS:
        if args.input_dir:
            raw = json.loads((args.input_dir / subset / "dataset.json").read_text())
        else:
            url = f"https://raw.githubusercontent.com/princeton-nlp/LLMBar/{args.revision}/Dataset/LLMBar/{subset}/dataset.json"
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = json.load(response)
        raw_counts[subset] = len(raw)
        for index, item in enumerate(raw):
            row = normalize_row(item, subset, index)
            key = exact_pair_key(row)
            reason = (
                "duplicate"
                if key in seen
                else "question/pair overlap"
                if key in overlap_pairs
                or normalize_whitespace(row["question"]) in overlap_questions
                else None
            )
            if reason:
                excluded.append({"pair_id": row["pair_id"], "reason": reason})
            else:
                selected.append(row)
                seen.add(key)
    write_jsonl(args.output, selected)
    write_json(
        args.report,
        {
            "revision": args.revision,
            "raw_counts": raw_counts,
            "written_total": len(selected),
            "question_groups": len({r["group_id"] for r in selected}),
            "exclusions": excluded,
            "reference_labels_used": False,
            "overlap_policy": str(args.exclude_overlap_with) if args.exclude_overlap_with else None,
        },
    )
    print(f"Wrote {len(selected)} LLMBar pairs")


if __name__ == "__main__":
    main()
