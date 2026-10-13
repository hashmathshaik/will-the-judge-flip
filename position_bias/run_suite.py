from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("judgebench", "llmbar", "mtbench"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--judges", nargs="+", help="Default: all eight judges")
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0"],
        help="Independent GPU IDs; one sequential task queue per GPU",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    models = json.loads(args.config.read_text())
    names = args.judges or list(models)
    if (
        len(set(names)) != len(names)
        or len(set(args.gpus)) != len(args.gpus)
        or set(names) - set(models)
    ):
        raise ValueError("Judge names and GPU IDs must be unique and judges must be configured")
    if args.device == "cpu" and len(args.gpus) != 1:
        raise ValueError("Use one worker for CPU execution")

    def worker(gpu, assigned):
        for name in assigned:
            model = models[name]
            directory = args.runs_dir / f"{name}_{args.benchmark}"
            if (directory / "judgments.jsonl").exists():
                raise FileExistsError(
                    f"Use a fresh runs directory: {directory} already contains judgments"
                )
            directory.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "position_bias.run_judge",
                "--input",
                str(args.input or Path(f"data/{args.benchmark}.jsonl")),
                "--output-dir",
                str(directory),
                "--model",
                model["model_id"],
                "--revision",
                model["revision"],
                "--device",
                args.device,
                "--dtype",
                "bfloat16" if args.device == "cuda" else "float32",
                "--context-limit",
                str(args.context_limit),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--samples",
                "0" if args.benchmark == "mtbench" else "5",
                "--extract-features",
            ]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print(
                f"GPU {gpu}: {name}/{args.benchmark}; log {directory / 'console.log'}", flush=True
            )
            with (directory / "console.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, names[i :: len(args.gpus)]) for i, gpu in enumerate(args.gpus)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
