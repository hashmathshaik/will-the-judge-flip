from __future__ import annotations

import unittest

import numpy as np

from position_bias.folds import add_flip_burden, make_grouped_folds
from position_bias.metrics import (
    bootstrap_ci,
    bootstrap_prediction_suite,
    paired_difference_ci,
    score,
    score_no_signal,
    stratified_auroc,
)
from position_bias.probe import nested_predict
from position_bias.run_analysis import baseline_feature_matrices
from position_bias.run_transfer import frozen_no_signal_predictions


class AnalysisPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "group_id": f"g{i}",
                "coarse_domain": ("knowledge", "math", "reasoning", "coding")[i % 4],
                "flip_burden_bin": ("low", "medium", "high")[i % 3],
            }
            for i in range(20)
        ]

    def test_grouped_folds_are_deterministic_and_cover_all_rows(self) -> None:
        first = make_grouped_folds(self.rows, n_splits=5, seed=11)
        second = make_grouped_folds(self.rows, n_splits=5, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(range(5)))
        counts = [first.count(fold) for fold in range(5)]
        self.assertLessEqual(max(counts) - min(counts), 1)
        for group in {row["group_id"] for row in self.rows}:
            values = {fold for row, fold in zip(self.rows, first) if row["group_id"] == group}
            self.assertEqual(len(values), 1)

    def test_metrics_and_group_bootstrap(self) -> None:
        y = np.asarray([0, 1, 0, 1, 1, 0])
        probabilities = np.asarray([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
        values = score(y, probabilities)
        self.assertAlmostEqual(values["auroc"], 1.0)
        self.assertLess(values["brier"], 0.1)
        groups = np.asarray(["a", "a", "b", "b", "c", "c"])
        interval = bootstrap_ci(y, probabilities, groups, repetitions=100, seed=3)
        self.assertEqual(interval["brier"]["valid_repetitions"], 100)
        difference = paired_difference_ci(
            y, probabilities, np.full(6, y.mean()), groups, repetitions=100, seed=3
        )
        self.assertEqual(difference["brier"]["valid_repetitions"], 100)

    def test_flip_burden_bin_uses_new_value_not_stale_row_value(self) -> None:
        row = {
            "group_id": "g",
            "coarse_domain": "math",
            "flip_burden": 4,
            "flip_burden_bin": "high",
        }
        updated = add_flip_burden([row], [0])[0]
        self.assertEqual(updated["flip_burden"], 0)
        self.assertEqual(updated["flip_burden_bin"], "low")

    def test_baselines_include_signed_and_absolute_features(self) -> None:
        features = baseline_feature_matrices(
            [
                {
                    "confidence": 80,
                    "signed_logit_margin": -3.5,
                    "signed_length_difference": -12,
                }
            ],
            ["X"],
        )
        np.testing.assert_array_equal(features["confidence"], [[80]])
        np.testing.assert_array_equal(features["logit_margin"], [[-3.5, 3.5]])
        np.testing.assert_array_equal(features["length_difference"], [[-12, 12]])

    def test_combined_baselines_concatenate_the_single_signal_columns(self) -> None:
        metadata = [
            {
                "confidence": 80,
                "signed_logit_margin": -3.5,
                "signed_length_difference": -12,
            },
            {
                "confidence": 60,
                "signed_logit_margin": 2.0,
                "signed_length_difference": 4,
            },
        ]
        features = baseline_feature_matrices(metadata, ["X", "Y"])
        np.testing.assert_array_equal(features["verdict_slot"], [[1.0], [0.0]])
        np.testing.assert_array_equal(
            features["combined_output"],
            [[80, -3.5, 3.5, -12, 12], [60, 2.0, 2.0, 4, 4]],
        )
        np.testing.assert_array_equal(
            features["combined_output_slot"],
            [[80, -3.5, 3.5, -12, 12, 1.0], [60, 2.0, 2.0, 4, 4, 0.0]],
        )

    def test_stratified_auroc_ignores_a_pure_stratum_separator(self) -> None:
        y = np.asarray([0, 1, 0, 1])
        strata = np.asarray(["X", "X", "Y", "Y"])
        separator = np.asarray([0.9, 0.9, 0.1, 0.1])
        self.assertEqual(stratified_auroc(y, separator, strata), 0.5)
        within = np.asarray([0.1, 0.9, 0.1, 0.9])
        self.assertEqual(stratified_auroc(y, within, strata), 1.0)

    def test_stratified_auroc_skips_a_stratum_with_one_class(self) -> None:
        y = np.asarray([0, 1, 1, 1])
        strata = np.asarray(["X", "X", "Y", "Y"])
        scores = np.asarray([0.2, 0.8, 0.4, 0.6])
        self.assertEqual(stratified_auroc(y, scores, strata), 1.0)

    def test_ood_no_signal_is_frozen_id_prevalence(self) -> None:
        predictions = frozen_no_signal_predictions(np.asarray([0, 1, 1, 0, 1]), 3)
        np.testing.assert_allclose(predictions, [0.6, 0.6, 0.6])

    def test_no_signal_discrimination_is_declared_chance(self) -> None:
        y = np.asarray([0, 1, 1, 0, 1])
        foldwise_prevalences = np.asarray([0.55, 0.45, 0.55, 0.45, 0.55])
        values = score_no_signal(y, foldwise_prevalences)
        self.assertEqual(values["auroc"], 0.5)
        self.assertEqual(values["auprc"], 0.6)

    def test_shared_bootstrap_keeps_no_signal_at_chance(self) -> None:
        y = np.asarray([0, 1, 0, 1, 1, 0])
        groups = np.asarray(["a", "a", "b", "b", "c", "c"])
        suite = bootstrap_prediction_suite(
            y,
            {
                "hidden": np.asarray([0.1, 0.9, 0.2, 0.8, 0.7, 0.3]),
                "no_signal": np.asarray([0.4, 0.6, 0.4, 0.6, 0.5, 0.5]),
            },
            groups,
            reference_name="hidden",
            no_signal_names={"no_signal"},
            repetitions=100,
            seed=5,
        )
        self.assertEqual(suite["predictors"]["no_signal"]["auroc"]["lower"], 0.5)
        self.assertEqual(suite["predictors"]["no_signal"]["auroc"]["upper"], 0.5)

    def test_nested_predict_can_score_a_second_feature_matrix(self) -> None:
        y = np.asarray([0, 1] * 10)
        features = (2 * y - 1).astype(float).reshape(-1, 1, 1)
        evaluation_features = -features
        outer_folds = [index // 4 for index in range(20)]
        rows = [{"group_id": f"g{index}"} for index in range(20)]

        def inner_builder(train_rows, _outer_fold):
            return [index % 2 for index in range(len(train_rows))]

        same, _ = nested_predict(features, y, outer_folds, inner_builder, rows, layer_numbers=[1])
        cross, _ = nested_predict(
            features,
            y,
            outer_folds,
            inner_builder,
            rows,
            layer_numbers=[1],
            evaluation_features=evaluation_features,
        )
        self.assertEqual(score(y, same)["auroc"], 1.0)
        self.assertEqual(score(y, cross)["auroc"], 0.0)


if __name__ == "__main__":
    unittest.main()
