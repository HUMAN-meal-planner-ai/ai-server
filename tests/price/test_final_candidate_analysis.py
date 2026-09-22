from __future__ import annotations

import unittest

import pandas as pd

from training.price.analyze_final_candidates import (
    _annotate_price_changes,
    _calculate_segment_metrics,
)


class FinalCandidateAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for model, predicted in (
            ("baseline_lag_1", [10.0, 10.0, 12.0, 10.0]),
            ("ridge", [10.0, 10.5, 14.0, 16.0]),
            ("gradient_boosting", [10.0, 11.0, 15.0, 19.0]),
        ):
            for offset, (actual, lag_1, prediction) in enumerate(
                zip([10.0, 11.0, 15.0, 20.0], [10.0, 10.0, 12.0, 10.0], predicted)
            ):
                rows.append(
                    {
                        "model": model,
                        "series_id": 1,
                        "price_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=offset),
                        "actual": actual,
                        "lag_1": lag_1,
                        "predicted": prediction,
                    }
                )
        self.predictions = pd.DataFrame(rows)

    def test_marks_changed_rows_and_uses_baseline_quantile_for_large_changes(self) -> None:
        annotated, threshold = _annotate_price_changes(self.predictions)

        baseline = annotated[annotated["model"] == "baseline_lag_1"]
        self.assertEqual(3, int(baseline["is_price_change"].sum()))
        self.assertEqual(1, int(baseline["is_large_change"].sum()))
        self.assertAlmostEqual(62.5, threshold)

    def test_segment_metrics_compare_the_same_rows_for_every_model(self) -> None:
        annotated, _ = _annotate_price_changes(self.predictions)
        metrics = _calculate_segment_metrics(annotated)

        counts = metrics.groupby("segment")["row_count"].nunique()
        self.assertTrue((counts == 1).all())
        changed = metrics[metrics["segment"] == "price_change"]
        self.assertTrue((changed["row_count"] == 3).all())


if __name__ == "__main__":
    unittest.main()
