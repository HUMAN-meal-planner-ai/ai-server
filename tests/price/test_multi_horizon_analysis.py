from __future__ import annotations

import unittest

import pandas as pd

from training.price.analyze_multi_horizon import (
    MULTI_HORIZON_TARGET,
    build_multi_horizon_frame,
    select_window_rows,
)
from training.price.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN


class MultiHorizonAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for offset in range(12):
            row = {
                "series_id": 3,
                "price_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=offset),
                TARGET_COLUMN: float(100 + offset),
            }
            for column in FEATURE_COLUMNS:
                row[column] = "WINTER" if column == "season" else float(offset + 1)
            rows.append(row)
        self.frame = pd.DataFrame(rows)

    def test_builds_calendar_day_targets_from_base_date_information(self) -> None:
        result = build_multi_horizon_frame(self.frame)
        base = result[result["base_date"] == pd.Timestamp("2026-01-01")]

        self.assertEqual(list(range(1, 8)), base["horizon"].tolist())
        self.assertEqual([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
                         base[MULTI_HORIZON_TARGET].tolist())
        self.assertTrue((base["current_price"] == 100.0).all())

    def test_future_price_change_does_not_change_base_features(self) -> None:
        original = build_multi_horizon_frame(self.frame)
        changed_source = self.frame.copy()
        changed_source.loc[
            changed_source["price_date"] == pd.Timestamp("2026-01-05"), TARGET_COLUMN
        ] = 999.0
        changed = build_multi_horizon_frame(changed_source)

        original_row = original[
            (original["base_date"] == pd.Timestamp("2026-01-01"))
            & (original["horizon"] == 4)
        ].iloc[0]
        changed_row = changed[
            (changed["base_date"] == pd.Timestamp("2026-01-01"))
            & (changed["horizon"] == 4)
        ].iloc[0]
        self.assertEqual(100.0, changed_row["current_price"])
        self.assertEqual(original_row["lag_1"], changed_row["lag_1"])
        self.assertNotEqual(
            original_row[MULTI_HORIZON_TARGET], changed_row[MULTI_HORIZON_TARGET]
        )

    def test_window_excludes_training_labels_from_evaluation_period(self) -> None:
        horizon_frame = build_multi_horizon_frame(self.frame)
        train, evaluation = select_window_rows(
            horizon_frame[horizon_frame["horizon"] == 3],
            train_target_end=pd.Timestamp("2026-01-06"),
            evaluation_start=pd.Timestamp("2026-01-07"),
            evaluation_end=pd.Timestamp("2026-01-11"),
        )

        self.assertLessEqual(train["target_date"].max(), pd.Timestamp("2026-01-06"))
        self.assertGreaterEqual(evaluation["base_date"].min(), pd.Timestamp("2026-01-07"))
        self.assertLessEqual(evaluation["target_date"].max(), pd.Timestamp("2026-01-11"))


if __name__ == "__main__":
    unittest.main()
