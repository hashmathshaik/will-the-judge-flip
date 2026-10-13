from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

UNCERTAINTY_COLUMNS = (
    "verdict_probability",
    "verdict_entropy",
    "label_entropy",
    "rationale_mean_log_probability",
    "rationale_mean_entropy",
)
BASELINE_NAMES = (
    "output",
    "uncertainty",
    "output_uncertainty",
    "input_tfidf",
    "input_embeddings",
    "rationale_tfidf",
    "rationale_embeddings",
    "all_visible",
    "sampling",
    "sampling_visible",
)
TFIDF_SETTINGS = {
    "max_features": 1024,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "lowercase": True,
}


def numeric_baselines(metadata: list[dict], records: list[dict]) -> dict[str, np.ndarray]:
    output = np.asarray(
        [
            [
                m["confidence"],
                m["signed_logit_margin"],
                abs(m["signed_logit_margin"]),
                m["response_a_tokens"],
                m["response_b_tokens"],
                m["signed_length_difference"],
                abs(m["signed_length_difference"]),
                m["rationale_tokens"],
                float(r["ab"]["parse"]["slot"] == "A"),
            ]
            for m, r in zip(metadata, records)
        ],
        dtype=float,
    )
    uncertainty = np.asarray([[m[c] for c in UNCERTAINTY_COLUMNS] for m in metadata], dtype=float)
    return {
        "output": output,
        "uncertainty": uncertainty,
        "output_uncertainty": np.hstack([output, uncertainty]),
    }


def sampling_features(records: list[dict]) -> np.ndarray:
    rows = []
    for record in records:
        samples = record.get("samples", [])
        if len(samples) != 5:
            raise ValueError("Sampling baselines require exactly five extra AB calls per pair")
        valid = [s for s in samples if s["parse"]["format_ok"]]
        counts = [sum(s["parse"]["slot"] == slot for s in valid) for slot in ("A", "B")]
        confidence = [s["parse"]["confidence"] for s in valid]
        rows.append(
            [
                min(counts) / len(valid) if valid else 0.0,
                sum(s["parse"]["slot"] != record["ab"]["parse"]["slot"] for s in valid) / len(valid)
                if valid
                else 0.0,
                float(np.mean(confidence)) if confidence else 0.0,
                float(np.std(confidence)) if confidence else 0.0,
                5 - len(valid),
            ]
        )
    return np.asarray(rows, dtype=float)


def text_hash(record: dict) -> str:
    text = record["input_text"] + "\0" + record["ab"]["parse"]["reasoning"]
    return hashlib.sha256(text.encode()).hexdigest()


def load_embeddings(path: Path, records: list[dict]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as file:
        ids = [str(x) for x in file["pair_ids"]]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate embedding pair IDs")
        lookup = {p: i for i, p in enumerate(ids)}
        indices = [lookup[r["pair_id"]] for r in records]
        hashes = [str(x) for x in file["text_hashes"]]
        if any(hashes[i] != text_hash(r) for i, r in zip(indices, records)):
            raise ValueError("Embeddings were built from different inputs or rationales")
        return {
            "input_embeddings": file["input_embeddings"][indices],
            "rationale_embeddings": file["rationale_embeddings"][indices],
        }


def all_baselines(
    metadata: list[dict], records: list[dict], embeddings: Path | None = None
) -> dict[str, np.ndarray]:
    matrices = numeric_baselines(metadata, records)
    for field in ("input", "rationale"):
        corpus = [
            r["input_text"] if field == "input" else r["ab"]["parse"]["reasoning"] for r in records
        ]

        vectorizer = TfidfVectorizer(**TFIDF_SETTINGS)
        matrices[field + "_tfidf"] = vectorizer.fit_transform(corpus).toarray()
    if embeddings is None:
        raise ValueError("The full ten-baseline comparison requires --embeddings LABEL=FILE.npz")
    matrices.update(load_embeddings(embeddings, records))
    matrices["all_visible"] = np.hstack(
        [
            matrices[n]
            for n in (
                "output",
                "uncertainty",
                "input_tfidf",
                "input_embeddings",
                "rationale_tfidf",
                "rationale_embeddings",
            )
        ]
    )
    matrices["sampling"] = sampling_features(records)
    matrices["sampling_visible"] = np.hstack([matrices["sampling"], matrices["output_uncertainty"]])
    if any(not np.isfinite(x).all() for x in matrices.values()):
        raise ValueError("Baseline matrix contains nonfinite values")
    return {name: matrices[name] for name in BASELINE_NAMES}


def main() -> None:
    import argparse

    from sentence_transformers import SentenceTransformer

    from position_bias.io_utils import read_jsonl, write_json

    parser = argparse.ArgumentParser(description="Extract pinned BGE input/rationale embeddings")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())["embeddings"]
    records = [r for r in read_jsonl(args.run / "judgments.jsonl") if r["ab"]["parse"]["format_ok"]]
    if not records:
        raise ValueError("No parse-valid AB judgments to embed")
    model = SentenceTransformer(
        config["model_id"],
        revision=config["revision"],
        device=args.device,
        cache_folder="data/hf_cache",
    )
    model.max_seq_length = config["max_seq_length"]
    encoded = {}
    for field in ("input", "rationale"):
        texts = [
            r["input_text"] if field == "input" else r["ab"]["parse"]["reasoning"] for r in records
        ]
        encoded[field + "_embeddings"] = model.encode(
            texts,
            batch_size=config["batch_size"],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pair_ids=np.asarray([r["pair_id"] for r in records]),
        text_hashes=np.asarray([text_hash(r) for r in records]),
        **encoded,
    )
    write_json(
        args.output.with_suffix(".json"),
        {
            "configuration": config,
            "n": len(records),
            "target_labels_used": False,
            "truncation": "first max_seq_length tokens",
        },
    )


if __name__ == "__main__":
    main()
