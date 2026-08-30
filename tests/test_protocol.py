from __future__ import annotations

import unittest

from position_bias.protocol import (
    build_user_prompt,
    candidate_layers,
    canonicalize,
    find_verdict_token_index,
    parse_completion,
)


class ProtocolTests(unittest.TestCase):
    def test_strict_parse(self) -> None:
        parsed = parse_completion(
            "REASONING: Response A is factually correct.\nCONFIDENCE: 87\nVERDICT: X"
        )
        self.assertTrue(parsed.format_ok)
        self.assertEqual(parsed.confidence, 87)
        self.assertEqual(parsed.slot, "X")

    def test_out_of_range_confidence_is_not_valid(self) -> None:
        parsed = parse_completion("REASONING: Short reason.\nCONFIDENCE: 101\nVERDICT: Y")
        self.assertFalse(parsed.format_ok)
        self.assertFalse(parsed.confidence_ok)
        self.assertTrue(parsed.verdict_reached)

    def test_extra_text_after_verdict_is_rejected(self) -> None:
        parsed = parse_completion("REASONING: Short reason.\nCONFIDENCE: 90\nVERDICT: Y\nThanks")
        self.assertFalse(parsed.format_ok)
        self.assertFalse(parsed.verdict_reached)

    def test_option_letter_verdicts_are_rejected(self) -> None:
        for letter in ("A", "B", "F", "H"):
            parsed = parse_completion(
                f"REASONING: Short reason.\nCONFIDENCE: 90\nVERDICT: {letter}"
            )
            self.assertFalse(parsed.format_ok, f"letter {letter} must not parse")
            self.assertIsNone(parsed.slot)

    def test_canonicalization_after_swap(self) -> None:
        self.assertEqual(canonicalize("X", "AB"), "A")
        self.assertEqual(canonicalize("Y", "AB"), "B")
        self.assertEqual(canonicalize("X", "BA"), "B")
        self.assertEqual(canonicalize("Y", "BA"), "A")

    def test_prompt_swap_changes_only_response_placement(self) -> None:
        pair = {"question": "Q", "response_a": "original A", "response_b": "original B"}
        ab = build_user_prompt(pair, "AB")
        ba = build_user_prompt(pair, "BA")
        self.assertLess(ab.index("original A"), ab.index("original B"))
        self.assertLess(ba.index("original B"), ba.index("original A"))
        self.assertEqual(ab.count("original A"), 1)
        self.assertEqual(ba.count("original A"), 1)

    def test_candidate_layers(self) -> None:
        self.assertEqual(candidate_layers(28), [7, 14, 21, 27])
        self.assertEqual(candidate_layers(36), [9, 18, 27, 35])

    def test_find_verdict_token_allows_trailing_whitespace(self) -> None:
        pieces = {1: "VERDICT:", 2: " X", 3: "\n"}

        def decode(ids: list[int]) -> str:
            return "".join(pieces[token] for token in ids)

        self.assertEqual(find_verdict_token_index([1, 2, 3], 2, decode), 1)


if __name__ == "__main__":
    unittest.main()
