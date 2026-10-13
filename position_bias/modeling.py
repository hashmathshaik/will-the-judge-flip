from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from position_bias.protocol import (
    ParseResult,
    candidate_layers,
    find_verdict_token_index,
    judge_messages,
    parse_completion,
)


@dataclass
class Runtime:
    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    layer_numbers: list[int]
    verdict_token_ids: dict[str, int]


@dataclass
class GeneratedJudgment:
    prompt_ids: list[int]
    completion_ids: list[int]
    completion: str
    parsed: ParseResult
    stopped_on_eos: bool
    extracted_features: ExtractedFeatures | None = None
    feature_error: str | None = None


@dataclass
class ExtractedFeatures:
    hidden_states: np.ndarray
    z_a: float
    z_b: float
    greedy_token_id: int
    recorded_verdict_token_id: int
    teacher_prefix_tokens: int
    input_hidden_states: np.ndarray
    uncertainty: dict[str, float]


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is visible")
    return device


def choose_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    choices = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if requested not in choices:
        raise ValueError(f"Unknown dtype: {requested}")
    if device.type == "cpu" and requested == "float16":
        raise ValueError("float16 CPU inference is unsupported; use float32 or bfloat16")
    return choices[requested]


def single_token_verdicts(tokenizer: Any) -> dict[str, int]:
    token_ids = {
        label: tokenizer.encode(f" {label}", add_special_tokens=False) for label in ("A", "B")
    }
    invalid = {label: ids for label, ids in token_ids.items() if len(ids) != 1}
    if invalid:
        raise ValueError(f"Verdict labels are not single tokens: {invalid}")
    return {label: ids[0] for label, ids in token_ids.items()}


def transformer_layers(model: Any) -> Sequence[Any]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError("Expected model.model.layers (Qwen, Llama, Granite, or Mistral)")
    return layers


def load_runtime(
    model_or_path: str,
    revision: str,
    cache_dir: Path,
    requested_device: str,
    requested_dtype: str,
    local_files_only: bool,
    cpu_threads: int,
    trust_remote_code: bool = False,
) -> Runtime:
    device = choose_device(requested_device)
    dtype = choose_dtype(requested_dtype, device)
    if device.type == "cpu":
        torch.set_num_threads(cpu_threads)

    tokenizer = AutoTokenizer.from_pretrained(
        model_or_path,
        revision=revision,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_or_path,
        revision=revision,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    layers = transformer_layers(model)
    return Runtime(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        layer_numbers=candidate_layers(len(layers)),
        verdict_token_ids=single_token_verdicts(tokenizer),
    )


def render_prompt_ids(runtime: Runtime, pair: dict[str, Any], order: str) -> list[int]:
    return runtime.tokenizer.apply_chat_template(
        judge_messages(pair, order),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_judgment(
    runtime: Runtime,
    pair: dict[str, Any],
    order: str,
    max_new_tokens: int,
    context_limit: int,
    sample: bool = False,
    temperature: float = 1.0,
    capture_features: bool = False,
) -> GeneratedJudgment:
    prompt_ids = render_prompt_ids(runtime, pair, order)
    if len(prompt_ids) + max_new_tokens > context_limit:
        raise ValueError(
            f"{pair['pair_id']} {order} needs {len(prompt_ids) + max_new_tokens} "
            f"tokens, above the frozen {context_limit}-token limit"
        )

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=runtime.device)
    attention_mask = torch.ones_like(input_ids)
    eos_token_ids = runtime.model.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = runtime.tokenizer.eos_token_id
    captured: dict[int, torch.Tensor] = {}
    history: list[tuple[np.ndarray, dict[str, float], int]] = []
    handles = []
    if capture_features:
        if sample:
            raise ValueError("Direct feature capture is for the primary greedy AB call")
        for number in runtime.layer_numbers:

            def capture(_module, _inputs, output, number=number):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[number] = hidden[:, -1, :].detach().float().cpu()

            handles.append(
                transformer_layers(runtime.model)[number - 1].register_forward_hook(capture)
            )

        def capture_logits(_module, _inputs, output):
            logits = output.logits[0, -1].float()
            log_probs = logits.log_softmax(dim=-1)
            greedy = int(logits.argmax().item())
            label_logits = logits[[runtime.verdict_token_ids["A"], runtime.verdict_token_ids["B"]]]
            label_log_probs = label_logits.log_softmax(dim=-1)
            label_probs = label_log_probs.exp()
            stats = {
                "z_a": float(label_logits[0].item()),
                "z_b": float(label_logits[1].item()),
                "greedy_log_probability": float(log_probs[greedy].item()),
                "verdict_probability": float(log_probs[greedy].exp().item()),
                "verdict_entropy": float(-(log_probs.exp() * log_probs).sum().item()),
                "label_entropy": float(-(label_probs * label_log_probs).sum().item()),
            }
            history.append(
                (
                    torch.cat([captured[n] for n in runtime.layer_numbers], dim=0).numpy(),
                    stats,
                    greedy,
                )
            )

        handles.append(runtime.model.register_forward_hook(capture_logits))
    try:
        with torch.inference_mode():
            sequence = runtime.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=sample,
                **({"temperature": temperature, "top_p": 1.0, "top_k": 0} if sample else {}),
                max_new_tokens=max_new_tokens,
                use_cache=True,
                num_beams=1,
                repetition_penalty=1.0,
                pad_token_id=runtime.tokenizer.pad_token_id
                if runtime.tokenizer.pad_token_id is not None
                else runtime.tokenizer.eos_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()
    completion_ids = sequence[0, len(prompt_ids) :].tolist()
    completion = runtime.tokenizer.decode(
        completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    eos_set = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or [])
    judgment = GeneratedJudgment(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion=completion,
        parsed=parse_completion(completion),
        stopped_on_eos=bool(completion_ids and completion_ids[-1] in eos_set),
    )
    if capture_features and judgment.parsed.format_ok:

        def decode(ids):
            return runtime.tokenizer.decode(
                list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        try:
            verdict_index = find_verdict_token_index(
                completion_ids, runtime.verdict_token_ids[judgment.parsed.slot], decode
            )
        except ValueError as exc:
            judgment.feature_error = str(exc)
            return judgment
        hidden, stats, greedy = history[verdict_index]
        if greedy != completion_ids[verdict_index]:
            raise RuntimeError(
                "Generation-time verdict logits disagree with the saved greedy verdict"
            )
        rationale_indices = [
            i
            for i in range(verdict_index)
            if len(decode(completion_ids[: i + 1])) <= judgment.parsed.rationale_end
        ]
        uncertainty = {
            name: stats[name]
            for name in ("verdict_probability", "verdict_entropy", "label_entropy")
        }
        uncertainty["rationale_mean_log_probability"] = (
            float(np.mean([history[i][1]["greedy_log_probability"] for i in rationale_indices]))
            if rationale_indices
            else 0.0
        )
        uncertainty["rationale_mean_entropy"] = (
            float(np.mean([history[i][1]["verdict_entropy"] for i in rationale_indices]))
            if rationale_indices
            else 0.0
        )
        judgment.extracted_features = ExtractedFeatures(
            hidden,
            stats["z_a"],
            stats["z_b"],
            greedy,
            completion_ids[verdict_index],
            len(prompt_ids) + verdict_index,
            history[0][0],
            uncertainty,
        )
    return judgment


def teacher_forced_extract(
    runtime: Runtime,
    judgment: GeneratedJudgment,
) -> ExtractedFeatures:
    if judgment.extracted_features is not None:
        return judgment.extracted_features
    if judgment.feature_error is not None:
        raise ValueError(judgment.feature_error)
    if not judgment.parsed.format_ok or judgment.parsed.slot is None:
        raise ValueError("Teacher-forced extraction requires a strictly parsed verdict")

    verdict_token_id = runtime.verdict_token_ids[judgment.parsed.slot]

    def decode(ids: list[int] | tuple[int, ...]) -> str:
        return runtime.tokenizer.decode(
            list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    verdict_index = find_verdict_token_index(judgment.completion_ids, verdict_token_id, decode)
    recorded_token_id = judgment.completion_ids[verdict_index]
    completion_prefix = judgment.completion_ids[:verdict_index]
    if not judgment.prompt_ids:
        raise ValueError("Empty teacher-forced prefix")

    captured: dict[int, torch.Tensor] = {}
    input_hidden_states: np.ndarray | None = None
    rationale_log_probs: list[float] = []
    rationale_entropies: list[float] = []
    rationale_end = judgment.parsed.rationale_end or 0
    handles = []
    layers = transformer_layers(runtime.model)
    for layer_number in runtime.layer_numbers:
        layer = layers[layer_number - 1]

        def capture(_module: Any, _inputs: Any, output: Any, number: int = layer_number) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[number] = hidden[:, -1, :].detach().float().cpu()

        handles.append(layer.register_forward_hook(capture))

    try:
        with torch.inference_mode():
            input_ids = torch.tensor([judgment.prompt_ids], dtype=torch.long, device=runtime.device)
            attention_mask = torch.ones_like(input_ids)
            forward_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            if "logits_to_keep" in inspect.signature(runtime.model.forward).parameters:
                forward_kwargs["logits_to_keep"] = 1
            outputs = runtime.model(**forward_kwargs)
            past_key_values = outputs.past_key_values
            input_hidden_states = torch.cat(
                [captured[number] for number in runtime.layer_numbers], dim=0
            ).numpy()

            for offset, token_id in enumerate(completion_prefix, start=1):
                if len(decode(completion_prefix[:offset])) <= rationale_end:
                    log_probs = outputs.logits[0, -1].float().log_softmax(dim=-1)
                    rationale_log_probs.append(float(log_probs[token_id].item()))
                    rationale_entropies.append(float(-(log_probs.exp() * log_probs).sum().item()))
                input_ids = torch.tensor([[token_id]], dtype=torch.long, device=runtime.device)
                attention_mask = torch.ones(
                    (1, len(judgment.prompt_ids) + offset),
                    dtype=torch.long,
                    device=runtime.device,
                )
                forward_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                if "logits_to_keep" in inspect.signature(runtime.model.forward).parameters:
                    forward_kwargs["logits_to_keep"] = 1
                outputs = runtime.model(**forward_kwargs)
                past_key_values = outputs.past_key_values
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(set(runtime.layer_numbers) - set(captured))
    if missing:
        raise RuntimeError(f"Hooks did not capture candidate layers: {missing}")
    final_logits = outputs.logits[0, -1].float()
    greedy_token_id = int(final_logits.argmax().item())
    if greedy_token_id != recorded_token_id:
        raise RuntimeError(
            "Teacher-forced greedy assertion failed: "
            f"recorded={recorded_token_id}, replay_argmax={greedy_token_id}"
        )

    hidden_states = torch.cat([captured[number] for number in runtime.layer_numbers], dim=0).numpy()
    label_logits = final_logits[[runtime.verdict_token_ids["A"], runtime.verdict_token_ids["B"]]]
    label_log_probs = label_logits.log_softmax(dim=-1)
    label_probs = label_log_probs.exp()
    verdict_log_probs = final_logits.log_softmax(dim=-1)
    uncertainty = {
        "verdict_probability": float(verdict_log_probs[recorded_token_id].exp().item()),
        "verdict_entropy": float(-(verdict_log_probs.exp() * verdict_log_probs).sum().item()),
        "label_entropy": float(-(label_probs * label_log_probs).sum().item()),
        "rationale_mean_log_probability": float(np.mean(rationale_log_probs))
        if rationale_log_probs
        else 0.0,
        "rationale_mean_entropy": float(np.mean(rationale_entropies))
        if rationale_entropies
        else 0.0,
    }
    if input_hidden_states is None:
        raise RuntimeError("Input-only state was not captured")
    return ExtractedFeatures(
        hidden_states=hidden_states,
        z_a=float(final_logits[runtime.verdict_token_ids["A"]].item()),
        z_b=float(final_logits[runtime.verdict_token_ids["B"]].item()),
        greedy_token_id=greedy_token_id,
        recorded_verdict_token_id=recorded_token_id,
        teacher_prefix_tokens=len(judgment.prompt_ids) + len(completion_prefix),
        input_hidden_states=input_hidden_states,
        uncertainty=uncertainty,
    )
