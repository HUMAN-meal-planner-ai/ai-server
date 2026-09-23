from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from training.price.analyze_weekly_forecast import (
    MAX_TARGET_COLUMN,
    RETURN_TARGET_COLUMN,
)
from training.price.compare_weekly_target_designs import (
    build_prediction_frame,
    target_returns,
)


class CompareWeeklyTargetDesignsTest(unittest.TestCase):
    def test_price_target_is_converted_to_return_against_current_price(self) -> None:
        frame = pd.DataFrame(
            {
                MAX_TARGET_COLUMN: [120.0],
                "current_price": [100.0],
            }
        )

        result = target_returns(frame, MAX_TARGET_COLUMN)

        np.testing.assert_allclose([0.2], result)

    def test_return_prediction_is_reconstructed_to_price_for_common_audit(self) -> None:
        evaluation = pd.DataFrame(
            {
                "series_id": [3],
                "ingredient_code": ["F00993"],
                "base_date": [pd.Timestamp("2026-01-01")],
                "current_price": [100.0],
            }
        )

        result = build_prediction_frame(
            RETURN_TARGET_COLUMN,
            "ridge",
            1,
            evaluation,
            np.asarray([0.2]),
            np.asarray([0.1]),
            0.15,
        ).iloc[0]

        self.assertEqual(120.0, result["actual_price"])
        self.assertAlmostEqual(110.0, result["predicted_price"])
        self.assertTrue(result["is_large_rise"])
        self.assertFalse(result["predicted_large_rise"])


if __name__ == "__main__":
    unittest.main()
