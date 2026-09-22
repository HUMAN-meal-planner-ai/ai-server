from __future__ import annotations

import unittest

from training.price.evaluation_metrics import calculate_metrics


class EvaluationMetricsTest(unittest.TestCase):
    def test_calculates_metrics_and_excludes_zero_from_mape(self) -> None:
        metrics = calculate_metrics([0.0, 10.0, 20.0], [1.0, 12.0, 18.0])

        self.assertEqual(1, metrics["mape_excluded_zero_count"])
        self.assertAlmostEqual(15.0, metrics["mape_percent"])
        self.assertIsNotNone(metrics["r2"])

    def test_constant_target_marks_r2_as_unavailable(self) -> None:
        metrics = calculate_metrics([10.0, 10.0], [9.0, 11.0])
        self.assertIsNone(metrics["r2"])
