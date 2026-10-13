from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Sequence, cast

Order = Literal["AB", "BA"]
Slot = Literal["A", "B"]
Canonical = Literal["A", "B"]

PROTOCOL_VERSION = "acl-order-instability-v5"
SYSTEM_PROMPT = (
    "Compare responses A and B for the request. Explain your comparison, "
    "report integer confidence from 0 to 100, and finish with exactly Verdict: A or Verdict: B."
    "\n\nUse exactly three lines, without markdown or extra text:\n"
    "Reasoning: <a concise comparison, at most 80 words, on one line>\n"
    "Confidence: <one integer from 0 to 100 expressing confidence in your chosen verdict>\n"
    "Verdict: <A or B>\n"
    "Replace the angle-bracket placeholders with your answer."
)

USER_TEMPLATE = """<USER_REQUEST>
{question}
</USER_REQUEST>

<RESPONSE_A>
{first_response}
</RESPONSE_A>

<RESPONSE_B>
{second_response}
</RESPONSE_B>"""

CONFIDENCE_LINE_RE = re.compile(
    r"(?i)(?:\*\*)?\b(?:The[ \t]+)?Confidence"
    r"(?:[ \t]+(?:level|score))?"
    r"(?:[ \t]+(?:in|for)[ \t]+(?:Response[ \t]+)?(?P<confidence_slot>[AB]|both[ \t]+responses))?"
    r"(?:\*\*)?[ \t]*(?::|=|[ \t]+is|[ \t]+of)[ \t]*(?:\*\*)?(?P<confidence>\d{1,3})(?![\d.])"
    r"(?:[ \t]*(?:%|/[ \t]*100|out[ \t]+of[ \t]+100))?"
)
INLINE_CONFIDENCE_RE = re.compile(
    r"(?i)\b(?:and[ \t]+)?(?:in|for)[ \t]+Response[ \t]+(?P<confidence_slot>[AB])"
    r"[ \t]*(?:is|:|=)[ \t]*(?P<confidence>\d{1,3})(?![\d.])"
)
VERDICT_SUFFIX_RE = re.compile(r"(?<!\S)Verdict:[ \t]*(?P<slot>[AB])\s*\Z")


@dataclass(frozen=True)
class ParseResult:
    format_ok: bool
    confidence_ok: bool
    verdict_reached: bool
    reasoning: str | None
    confidence: int | None
    slot: Slot | None
    error: str | None
    rationale_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_user_prompt(pair: dict[str, Any], order: Order) -> str:
    if order == "AB":
        first, second = pair["response_a"], pair["response_b"]
    elif order == "BA":
        first, second = pair["response_b"], pair["response_a"]
    else:
        raise ValueError(f"Unknown order: {order}")
    return USER_TEMPLATE.format(
        question=pair["question"], first_response=first, second_response=second
    )


def judge_messages(pair: dict[str, Any], order: Order) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(pair, order)},
    ]


def parse_completion(text: str) -> ParseResult:
    confidence_matches = list(CONFIDENCE_LINE_RE.finditer(text))

    for match in INLINE_CONFIDENCE_RE.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        if re.search(r"(?i)\bconfidence\b", text[line_start : match.start()]):
            if not any(
                match.span("confidence") == old.span("confidence") for old in confidence_matches
            ):
                confidence_matches.append(match)
    verdict_match = VERDICT_SUFFIX_RE.search(text)
    eligible_confidence = [
        match
        for match in confidence_matches
        if verdict_match
        and match.end() <= verdict_match.start()
        and (
            match.group("confidence_slot") is None
            or match.group("confidence_slot").lower().startswith("both")
            or match.group("confidence_slot").upper() == verdict_match.group("slot")
        )
    ]
    confidence_match = eligible_confidence[0] if len(eligible_confidence) == 1 else None

    confidence: int | None = None
    confidence_ok = False
    if confidence_match:
        confidence = int(confidence_match.group("confidence"))
        confidence_ok = 0 <= confidence <= 100

    slot: Slot | None = None
    if verdict_match:
        slot = cast(Slot, verdict_match.group("slot"))

    rationale_end = min(match.start() for match in confidence_matches) if confidence_match else None
    reasoning = text[:rationale_end].strip() if rationale_end is not None else ""
    ordered = bool(
        confidence_match and verdict_match and confidence_match.end() <= verdict_match.start()
    )
    if confidence_ok and verdict_match and reasoning and ordered:
        return ParseResult(
            format_ok=True,
            confidence_ok=True,
            verdict_reached=True,
            reasoning=re.sub(r"(?i)^REASONING:\s*", "", reasoning),
            confidence=confidence,
            slot=cast(Slot, verdict_match.group("slot")),
            error=None,
            rationale_end=rationale_end,
        )

    errors: list[str] = []
    if not verdict_match:
        errors.append("missing final Verdict: A/B line")
    if not confidence_match:
        errors.append("missing or ambiguous explicit confidence for the selected response")
    elif not confidence_ok:
        errors.append("confidence outside 0..100")
    if verdict_match and confidence_ok:
        errors.append("missing rationale or malformed confidence/verdict suffix")
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
    if slot not in ("A", "B"):
        raise ValueError(f"Unknown displayed slot: {slot}")
    if order == "AB":
        return slot
    if order == "BA":
        return "B" if slot == "A" else "A"
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
        if re.search(r"Verdict:\s*\Z", prefix):
            return index
    raise ValueError("Could not locate a single-token final verdict value")
