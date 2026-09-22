from __future__ import annotations

import unittest

import pandas as pd

from training.price.feature_engineering import FEATURE_COLUMNS, build_price_features


class PriceFeatureEngineeringTest(unittest.TestCase):
    def test_series_are_independent_and_maximum_lag_removes_first_thirty_rows(self) -> None:
        first = self._series(1, 100.0, periods=35)
        second = self._series(2, 1_000.0, periods=35)

        result = build_price_features(pd.concat([second, first], ignore_index=True))

        self.assertEqual([1, 2], result["series_id"].drop_duplicates().tolist())
        self.assertEqual(10, len(result))
        first_output = result[result["series_id"] == 1].iloc[0]
        second_output = result[result["series_id"] == 2].iloc[0]
        self.assertEqual(129.0, first_output["lag_1"])
        self.assertEqual(1_029.0, second_output["lag_1"])

    def test_current_target_is_not_included_in_rolling_features(self) -> None:
        prices = self._series(1, 1.0, periods=35)
        prices.loc[30, "standard_unit_price"] = 100_000.0

        result = build_price_features(prices)
        row = result[result["price_date"] == pd.Timestamp("2026-01-31")].iloc[0]

        self.assertEqual(100_000.0, row["target_standard_unit_price"])
        self.assertEqual(27.0, row["rolling_mean_7"])
        self.assertNotEqual(100_000.0, row["lag_1"])

    def test_changing_current_target_does_not_change_any_feature_on_same_date(self) -> None:
        original = self._series(1, 1.0, periods=35)
        changed = original.copy()
        changed.loc[30, "standard_unit_price"] = 100_000.0

        original_features = build_price_features(original).iloc[0]
        changed_features = build_price_features(changed).iloc[0]

        pd.testing.assert_series_equal(
            original_features[list(FEATURE_COLUMNS)],
            changed_features[list(FEATURE_COLUMNS)],
        )
        self.assertNotEqual(
            original_features["target_standard_unit_price"],
            changed_features["target_standard_unit_price"],
        )

    def test_lags_follow_observations_without_calendar_fill(self) -> None:
        prices = self._series(1, 10.0, periods=35)
        prices.loc[30:, "price_date"] = pd.date_range("2026-03-05", periods=5, freq="D")

        result = build_price_features(prices)
        first_output = result.iloc[0]

        self.assertEqual(34.0, first_output["days_since_previous"])
        self.assertEqual(39.0, first_output["lag_1"])
        self.assertFalse(result[list(FEATURE_COLUMNS)].isna().any().any())

    def test_calendar_features_and_season_are_derived_from_target_date(self) -> None:
        prices = self._series(1, 1.0, periods=35)
        result = build_price_features(prices)
        first_output = result.iloc[0]

        self.assertEqual(5, first_output["day_of_week"])
        self.assertEqual(1, first_output["month"])
        self.assertEqual("WINTER", first_output["season"])

    def test_duplicate_series_date_is_rejected(self) -> None:
        prices = self._series(1, 1.0, periods=35)
        duplicate = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "중복"):
            build_price_features(duplicate)

    def _series(self, series_id: int, start_price: float, periods: int) -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=periods, freq="D")
        values = [start_price + index for index in range(periods)]
        return pd.DataFrame(
            {
                "series_id": series_id,
                "price_date": dates,
                "original_price": [value * 1_000 for value in values],
                "standard_unit_price": values,
            }
        )


if __name__ == "__main__":
    unittest.main()
