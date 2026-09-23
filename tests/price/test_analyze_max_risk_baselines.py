from __future__ import annotations

import unittest

import numpy as np

from training.price.analyze_max_risk_baselines import binary_metrics


class AnalyzeMaxRiskBaselinesTest(unittest.TestCase):
    def test_binary_metrics_include_confusion_matrix_and_warning_rate(self) -> None:
        result = binary_metrics(
            np.asarray([True, True, False, False]),
            np.asarray([True, False, True, False]),
        )

        self.assertEqual(1, result["true_positive"])
        self.assertEqual(1, result["false_positive"])
        self.assertEqual(1, result["true_negative"])
        self.assertEqual(1, result["false_negative"])
        self.assertEqual(50.0, result["balanced_accuracy_percent"])
        self.assertEqual(50.0, result["warning_rate_percent"])


if __name__ == "__main__":
    unittest.main()
