from __future__ import annotations

import unittest

from position_bias.prepare_judgebench import build_group_id, exact_pair_key


class DataTests(unittest.TestCase):
    def test_exact_pair_key_catches_reversed_duplicates(self) -> None:
        first = {"question": "Q", "response_a": "one", "response_b": "two"}
        second = {"question": "Q", "response_a": "two", "response_b": "one"}
        self.assertEqual(exact_pair_key(first), exact_pair_key(second))

    def test_null_original_id_does_not_collapse_groups(self) -> None:
        first = build_group_id("livebench-reasoning", None, "question one")
        second = build_group_id("livebench-reasoning", None, "question two")
        self.assertNotEqual(first, second)

    def test_group_id_merges_the_same_question_across_splits(self) -> None:
        gpt = build_group_id("livebench-math", None, "What is 2 + 2?")
        claude = build_group_id("livebench-math", None, "What is 2 + 2?\n")
        self.assertEqual(gpt, claude)

    def test_group_id_uses_original_id_when_present(self) -> None:
        self.assertEqual(build_group_id("mmlu-pro-law", 1420, "q"), "mmlu-pro-law::1420")


if __name__ == "__main__":
    unittest.main()
