from __future__ import annotations

import unittest

import pandas as pd

from training.price.analyze_weekly_forecast import TARGET_COLUMN
from training.price.build_regional_weekly_dataset import build_regional_weekly_dataset


class RegionalWeeklyDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = pd.DataFrame(
            {
                "series_id": [3] * 45,
                "ingredient_code": ["F00993"] * 45,
                "price_date": pd.date_range("2026-01-01", periods=45, freq="D"),
                "representative_standard_unit_price": [
                    float(100 + index) for index in range(45)
                ],
            }
        )

    def test_target_uses_next_seven_days_and_keeps_ingredient_identity(self) -> None:
        result = build_regional_weekly_dataset(self.prices)
        row = result[result["base_date"] == pd.Timestamp("2026-01-29")].iloc[0]

        self.assertEqual("F00993", row["ingredient_code"])
        self.assertEqual(132.0, row[TARGET_COLUMN])
        self.assertEqual(7, row["future_observation_count"])

    def test_future_change_does_not_change_features_at_base_date(self) -> None:
        changed_prices = self.prices.copy()
        changed_prices.loc[
            changed_prices["price_date"] > "2026-01-29",
            "representative_standard_unit_price",
        ] += 500

        original = build_regional_weekly_dataset(self.prices)
        changed = build_regional_weekly_dataset(changed_prices)
        original_row = original[original["base_date"] == pd.Timestamp("2026-01-29")].iloc[0]
        changed_row = changed[changed["base_date"] == pd.Timestamp("2026-01-29")].iloc[0]

        self.assertEqual(original_row["mean_28d"], changed_row["mean_28d"])
        self.assertEqual(original_row["lag_28d"], changed_row["lag_28d"])
        self.assertNotEqual(original_row[TARGET_COLUMN], changed_row[TARGET_COLUMN])


if __name__ == "__main__":
    unittest.main()
