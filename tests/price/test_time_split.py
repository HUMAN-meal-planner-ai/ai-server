from __future__ import annotations

import unittest

import pandas as pd

from training.price.time_split import determine_time_boundaries, split_name_for_dates


class TimeSplitTest(unittest.TestCase):
    def test_uses_global_chronological_boundaries_without_date_overlap(self) -> None:
        dates = pd.date_range("2026-01-01", periods=20, freq="D")
        frame = pd.DataFrame(
            {
                "series_id": [1] * 20 + [2] * 20,
                "price_date": list(dates) + list(dates),
            }
        )

        boundaries = determine_time_boundaries(frame)
        split = split_name_for_dates(frame["price_date"], boundaries)

        self.assertEqual(pd.Timestamp("2026-01-14"), boundaries.train_end)
        self.assertEqual(pd.Timestamp("2026-01-17"), boundaries.validation_end)
        self.assertTrue((frame.loc[split == "train", "price_date"] <= boundaries.train_end).all())
        self.assertTrue((frame.loc[split == "test", "price_date"] > boundaries.validation_end).all())
