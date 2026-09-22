from __future__ import annotations

import unittest

import pandas as pd

from training.price.analyze_weekly_forecast import (
    TARGET_COLUMN,
    annotate_predictions,
    build_weekly_forecast_frame,
    select_window_rows,
)


class WeeklyForecastAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = pd.DataFrame(
            {
                "series_id": [3] * 45,
                "price_date": pd.date_range("2026-01-01", periods=45, freq="D"),
                "standard_unit_price": [float(100 + index) for index in range(45)],
            }
        )

    def test_target_is_mean_of_only_the_next_seven_calendar_days(self) -> None:
        result = build_weekly_forecast_frame(self.prices)
        row = result[result["base_date"] == pd.Timestamp("2026-01-29")].iloc[0]

        self.assertEqual(128.0, row["current_price"])
        self.assertEqual(132.0, row[TARGET_COLUMN])
        self.assertEqual(7, row["future_observation_count"])

    def test_future_change_does_not_change_base_features(self) -> None:
        original = build_weekly_forecast_frame(self.prices)
        changed_prices = self.prices.copy()
        changed_prices.loc[changed_prices["price_date"] > "2026-01-29", "standard_unit_price"] += 500
        changed = build_weekly_forecast_frame(changed_prices)

        original_row = original[original["base_date"] == pd.Timestamp("2026-01-29")].iloc[0]
        changed_row = changed[changed["base_date"] == pd.Timestamp("2026-01-29")].iloc[0]
        self.assertEqual(original_row["mean_28d"], changed_row["mean_28d"])
        self.assertEqual(original_row["current_price"], changed_row["current_price"])
        self.assertNotEqual(original_row[TARGET_COLUMN], changed_row[TARGET_COLUMN])

    def test_training_target_window_ends_before_evaluation(self) -> None:
        frame = build_weekly_forecast_frame(self.prices)
        train, evaluation = select_window_rows(
            frame,
            train_target_end=pd.Timestamp("2026-02-05"),
            evaluation_start=pd.Timestamp("2026-02-01"),
            evaluation_end=pd.Timestamp("2026-02-12"),
        )

        self.assertLessEqual(
            train["target_end_date"].max(), pd.Timestamp("2026-02-05")
        )
        self.assertGreaterEqual(
            evaluation["base_date"].min(), pd.Timestamp("2026-02-01")
        )

    def test_drops_right_censored_weekly_target(self) -> None:
        frame = build_weekly_forecast_frame(self.prices)

        self.assertEqual(pd.Timestamp("2026-02-07"), frame["base_date"].max())
        self.assertEqual(pd.Timestamp("2026-02-14"), frame["target_end_date"].max())

    def test_persistence_cannot_detect_non_flat_direction_or_large_change(self) -> None:
        predictions = pd.DataFrame(
            {
                "fold": [1],
                "model": ["baseline_lag_1"],
                "series_id": [3],
                "base_date": [pd.Timestamp("2026-01-01")],
                "target_end_date": [pd.Timestamp("2026-01-08")],
                "current_price": [100.0],
                "actual": [120.0],
                "predicted": [100.0],
                "large_change_threshold": [0.1],
            }
        )

        annotated = annotate_predictions(predictions).iloc[0]

        self.assertFalse(annotated["direction_correct"])
        self.assertTrue(annotated["is_large_change"])
        self.assertFalse(annotated["predicted_large_change"])


if __name__ == "__main__":
    unittest.main()
