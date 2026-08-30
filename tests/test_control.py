from __future__ import annotations

import unittest

from position_bias.prepare_control import select_control_rows
from position_bias.run_control import summarize
from position_bias.summarize_control import aggregate


class ControlTests(unittest.TestCase):
    def test_subset_selection_is_deterministic_and_ignores_row_order(self) -> None:
        dataset = [{"pair_id": f"p{i}", "value": i} for i in range(10)]
        common = [{"pair_id": f"p{i}", "labels": {"judge": i % 2}} for i in range(10)]
        first = select_control_rows(dataset, common, size=4, seed=7)
        second = select_control_rows(list(reversed(dataset)), list(reversed(common)), 4, 7)
        self.assertEqual(first, second)
        self.assertEqual(len({row["pair_id"] for row in first}), 4)

    def test_control_summary_uses_only_parse_valid_ab2(self) -> None:
        records = [
            {
                "ab1": {"format_ok": True},
                "ba": {"format_ok": True},
                "ab2": {"parse": {"format_ok": True}},
                "same_order_disagreement": 0,
                "position_flip": 1,
            },
            {
                "ab1": {"format_ok": True},
                "ba": {"format_ok": True},
                "ab2": {"parse": {"format_ok": False}},
                "same_order_disagreement": None,
                "position_flip": 0,
            },
        ]
        result = summarize(records, requested=2)
        self.assertEqual(result["same_order_usable"], 1)
        self.assertEqual(result["same_order_disagreements"], 0)
        self.assertEqual(result["swap_usable"], 2)
        self.assertEqual(result["swap_disagreements"], 1)

    def test_aggregate_rejects_different_subsets(self) -> None:
        complete = {
            "subset_pair_ids": ["a", "b"],
            "summary": {"pairs_completed": 2, "pairs_requested": 2},
        }
        other = {
            "subset_pair_ids": ["a", "c"],
            "summary": {"pairs_completed": 2, "pairs_requested": 2},
        }
        with self.assertRaises(ValueError):
            aggregate({"first": complete, "second": other})


if __name__ == "__main__":
    unittest.main()
