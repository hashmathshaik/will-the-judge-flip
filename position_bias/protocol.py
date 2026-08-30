from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Sequence, cast

Order = Literal["AB", "BA"]
Slot = Literal["X", "Y"]
Canonical = Literal["A", "B"]

SYSTEM_PROMPT = """You are an impartial evaluator of two candidate responses.
Judge correctness, relevance, reasoning quality, and instruction following. Prefer
substantive quality over style or response length. You must choose one response;
ties are not allowed. Follow the requested output format exactly."""

USER_TEMPLATE = """Evaluate the two responses to the user request below.

<USER_REQUEST>
{question}
</USER_REQUEST>

<RESPONSE_X>
{response_x}
</RESPONSE_X>

<RESPONSE_Y>
{response_y}
</RESPONSE_Y>

Return exactly these three lines and nothing else:
REASONING: one short paragraph
CONFIDENCE: one integer from 0 to 100
VERDICT: X or Y"""

STRICT_OUTPUT_RE = re.compile(
    r"\A\s*REASONING:\s*(?P<reasoning>.+?)\s*\n"
    r"CONFIDENCE:\s*(?P<confidence>\d{1,3})\s*\n"
    r"VERDICT:\s*(?P<slot>[XY])\s*\Z",
    flags=re.DOTALL,
)
CONFIDENCE_LINE_RE = re.compile(r"(?m)^CONFIDENCE:\s*(?P<confidence>\d{1,3})\s*$")
VERDICT_SUFFIX_RE = re.compile(r"(?m)^VERDICT:\s*(?P<slot>[XY])\s*\Z")


@dataclass(frozen=True)
class ParseResult:
    format_ok: bool
    confidence_ok: bool
    verdict_reached: bool
    reasoning: str | None
    confidence: int | None
    slot: Slot | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_user_prompt(pair: dict[str, Any], order: Order) -> str:
    if order == "AB":
        first, second = pair["response_a"], pair["response_b"]
    elif order == "BA":
        first, second = pair["response_b"], pair["response_a"]
    else:
        raise ValueError(f"Unknown order: {order}")
    return USER_TEMPLATE.format(question=pair["question"], response_x=first, response_y=second)


def judge_messages(pair: dict[str, Any], order: Order) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(pair, order)},
    ]


def parse_completion(text: str) -> ParseResult:
    strict = STRICT_OUTPUT_RE.fullmatch(text)
    confidence_match = CONFIDENCE_LINE_RE.search(text)
    verdict_match = VERDICT_SUFFIX_RE.search(text)

    confidence: int | None = None
    confidence_ok = False
    if confidence_match:
        confidence = int(confidence_match.group("confidence"))
        confidence_ok = 0 <= confidence <= 100

    slot: Slot | None = None
    if verdict_match:
        slot = cast(Slot, verdict_match.group("slot"))

    if strict and confidence_ok:
        return ParseResult(
            format_ok=True,
            confidence_ok=True,
            verdict_reached=True,
            reasoning=strict.group("reasoning").strip(),
            confidence=confidence,
            slot=cast(Slot, strict.group("slot")),
            error=None,
        )

    errors: list[str] = []
    if not verdict_match:
        errors.append("missing final VERDICT line")
    if not confidence_match:
        errors.append("missing CONFIDENCE line")
    elif not confidence_ok:
        errors.append("confidence outside 0..100")
    if strict is None and verdict_match and confidence_ok:
        errors.append("output does not match exact three-line format")
    return ParseResult(
        format_ok=False,
        confidence_ok=confidence_ok,
        verdict_reached=verdict_match is not None,
        reasoning=None,
        confidence=confidence,
        slot=slot,
        error="; ".join(errors) or "invalid output",
    )


def canonicalize(slot: Slot, order: Order) -> Canonical:
    if order == "AB":
        return "A" if slot == "X" else "B"
    if order == "BA":
        return "B" if slot == "X" else "A"
    raise ValueError(f"Unknown order: {order}")


def coarse_domain(source: str) -> str:
    normalized = source.lower().replace("_", "-")
    if "code" in normalized:
        return "coding"
    if "math" in normalized:
        return "math"
    if "reason" in normalized:
        return "reasoning"
    return "knowledge"


def candidate_layers(number_of_layers: int) -> list[int]:
    if number_of_layers < 2:
        raise ValueError("At least two transformer layers are required")
    indices = {
        math.ceil(0.25 * number_of_layers),
        math.ceil(0.50 * number_of_layers),
        math.ceil(0.75 * number_of_layers),
        number_of_layers - 1,
    }
    return sorted(indices)


def find_verdict_token_index(
    token_ids: Sequence[int],
    verdict_token_id: int,
    decode: Callable[[Sequence[int]], str],
) -> int:
    for index in range(len(token_ids) - 1, -1, -1):
        if token_ids[index] != verdict_token_id:
            continue
        if decode(token_ids[index + 1 :]).strip():
            continue
        prefix = decode(token_ids[:index])
        if re.search(r"VERDICT:\s*\Z", prefix):
            return index
    raise ValueError("Could not locate a single-token final verdict value")
