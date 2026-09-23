from __future__ import annotations

import unittest

import pandas as pd

from training.price.compare_regional_weekly_models import (
    build_expanding_folds,
    calculate_direction_metrics,
    calculate_risk_metrics,
    select_fold_rows,
)


class CompareRegionalWeeklyModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        self.frame = pd.DataFrame(
            {
                "series_id": [3] * 100,
                "base_date": dates,
                "target_end_date": dates + pd.Timedelta(days=7),
            }
        )

    def test_folds_are_chronological_and_non_overlapping(self) -> None:
        folds = build_expanding_folds(self.frame, fold_count=3, initial_train_fraction=0.7)

        self.assertEqual(3, len(folds))
        self.assertLess(folds[0][1], folds[0][2])
        self.assertLess(folds[0][2], folds[1][1])
        self.assertLess(folds[1][2], folds[2][1])

    def test_training_target_finishes_before_evaluation_starts(self) -> None:
        evaluation_start = pd.Timestamp("2024-03-20")
        evaluation_end = pd.Timestamp("2024-04-01")

        train, evaluation = select_fold_rows(
            self.frame, evaluation_start, evaluation_end
        )

        self.assertLess(train["target_end_date"].max(), evaluation_start)
        self.assertGreaterEqual(evaluation["base_date"].min(), evaluation_start)
        self.assertLessEqual(evaluation["base_date"].max(), evaluation_end)

    def test_direction_metrics_separate_rises_and_falls(self) -> None:
        predictions = pd.DataFrame(
            {
                "model": ["ridge", "ridge"],
                "actual_change": [0.1, -0.1],
                "is_change": [True, True],
                "direction_correct": [True, False],
            }
        )

        result = calculate_direction_metrics(predictions)

        rise = result[result["direction"] == "rise"].iloc[0]
        fall = result[result["direction"] == "fall"].iloc[0]
        self.assertEqual(100.0, rise["accuracy_percent"])
        self.assertEqual(0.0, fall["accuracy_percent"])

    def test_large_rise_risk_is_evaluated_separately(self) -> None:
        predictions = pd.DataFrame(
            {
                "model": ["ridge", "ridge"],
                "actual_change": [0.2, -0.2],
                "predicted_change": [0.2, 0.2],
                "large_change_threshold": [0.1, 0.1],
                "is_large_change": [True, True],
                "predicted_large_change": [True, True],
            }
        )

        result = calculate_risk_metrics(predictions)
        large_rise = result[result["risk_type"] == "large_rise"].iloc[0]

        self.assertEqual(1, large_rise["true_positive"])
        self.assertEqual(1, large_rise["false_positive"])
        self.assertEqual(50.0, large_rise["precision_percent"])


if __name__ == "__main__":
    unittest.main()
