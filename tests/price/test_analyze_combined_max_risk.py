from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from training.price.analyze_combined_max_risk import (
    calculate_top_n_metrics,
    empirical_percentile,
)


class AnalyzeCombinedMaxRiskTest(unittest.TestCase):
    def test_empirical_percentile_uses_only_reference_distribution(self) -> None:
        result = empirical_percentile(
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            np.asarray([0.0, 2.0, 5.0]),
        )

        np.testing.assert_allclose([0.0, 0.5, 1.0], result)

    def test_top_n_metrics_are_calculated_per_date(self) -> None:
        frame = pd.DataFrame(
            {
                "strategy": ["combined"] * 6,
                "base_date": pd.to_datetime(
                    ["2026-01-01"] * 3 + ["2026-01-02"] * 3
                ),
                "actual_large_rise": [True, False, False, False, True, True],
                "ranking_score": [0.9, 0.8, 0.1, 0.9, 0.8, 0.1],
            }
        )

        result = calculate_top_n_metrics(frame, ["strategy"])
        top_five = result[result["top_n"] == 5].iloc[0]

        self.assertEqual(2, top_five["date_count"])
        self.assertEqual(2, top_five["dates_with_large_rise"])
        self.assertAlmostEqual(50.0, top_five["micro_precision_at_n_percent"])
        self.assertAlmostEqual(100.0, top_five["micro_recall_at_n_percent"])


if __name__ == "__main__":
    unittest.main()
