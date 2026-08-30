from __future__ import annotations

import unittest

from position_bias.prepare_mtbench import normalize_row


class MtBenchPreparationTests(unittest.TestCase):
    def make_row(self, turn: int = 2) -> dict:
        return {
            "question_id": 81,
            "model_a": "z-model",
            "model_b": "a-model",
            "turn": turn,
            "conversation_a": [
                {"role": "user", "content": "first request"},
                {"role": "assistant", "content": "z first"},
                {"role": "user", "content": "revise it"},
                {"role": "assistant", "content": "z second"},
            ],
            "conversation_b": [
                {"role": "user", "content": "first request"},
                {"role": "assistant", "content": "a first"},
                {"role": "user", "content": "revise it"},
                {"role": "assistant", "content": "a second"},
            ],
        }

    def test_turn_two_preserves_candidate_specific_history(self) -> None:
        row = normalize_row(self.make_row())
        self.assertIn("USER TURN 1", row["question"])
        self.assertIn("USER TURN 2", row["question"])
        self.assertEqual(row["model_a"], "a-model")
        self.assertIn("a first", row["response_a"])
        self.assertIn("a second", row["response_a"])

    def test_reversing_annotation_order_keeps_canonical_pair_id(self) -> None:
        original = self.make_row()
        reversed_row = {
            **original,
            "model_a": original["model_b"],
            "model_b": original["model_a"],
            "conversation_a": original["conversation_b"],
            "conversation_b": original["conversation_a"],
        }
        self.assertEqual(normalize_row(original)["pair_id"], normalize_row(reversed_row)["pair_id"])


if __name__ == "__main__":
    unittest.main()
