from __future__ import annotations

import unittest

import numpy as np

from training.price.analyze_combined_risk_thresholds import (
    select_threshold_for_recall,
)


class AnalyzeCombinedRiskThresholdsTest(unittest.TestCase):
    def test_selects_highest_threshold_that_meets_target_recall(self) -> None:
        scores = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        actual = np.asarray([False, True, True, True, True, False])

        threshold = select_threshold_for_recall(scores, actual, 0.75)

        self.assertEqual(0.5, threshold)
        self.assertGreaterEqual((scores[actual] >= threshold).mean(), 0.75)

    def test_rejects_invalid_target_recall(self) -> None:
        with self.assertRaisesRegex(ValueError, "목표 Recall"):
            select_threshold_for_recall(
                np.asarray([0.5]), np.asarray([True]), 0.0
            )


if __name__ == "__main__":
    unittest.main()
