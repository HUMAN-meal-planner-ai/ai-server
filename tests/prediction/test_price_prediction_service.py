from __future__ import annotations

import unittest
from datetime import UTC, date, datetime
from decimal import Decimal

from app.prediction.repository import LatestSeriesPrice
from app.prediction.service import PricePredictionService, PriceSeriesNotFoundError


class StubRepository:
    def __init__(self, prices: list[LatestSeriesPrice]) -> None:
        self.prices = prices

    def find_latest_prices(self, series_ids: list[int]) -> list[LatestSeriesPrice]:
        return self.prices


class PricePredictionServiceTest(unittest.TestCase):
    def test_uses_latest_actual_price_as_next_prediction(self) -> None:
        latest = LatestSeriesPrice(
            series_id=3,
            base_date=date(2026, 9, 18),
            base_price=Decimal("0.820000"),
            standard_unit="g",
        )
        now = datetime(2026, 9, 22, 3, 0, tzinfo=UTC)
        service = PricePredictionService(StubRepository([latest]), clock=lambda: now)

        result = service.predict_next([3])[0]

        self.assertEqual(latest.base_price, result.base_price)
        self.assertEqual(latest.base_price, result.predicted_price)
        self.assertEqual(date(2026, 9, 19), result.target_date)
        self.assertEqual(now, result.generated_at)

    def test_fails_when_any_requested_series_has_no_price(self) -> None:
        service = PricePredictionService(StubRepository([]))

        with self.assertRaisesRegex(PriceSeriesNotFoundError, "3, 17"):
            service.predict_next([3, 17])

    def test_returns_seven_daily_predictions_from_latest_actual_price(self) -> None:
        latest = LatestSeriesPrice(
            series_id=3,
            base_date=date(2026, 9, 18),
            base_price=Decimal("0.820000"),
            standard_unit="g",
        )
        now = datetime(2026, 9, 22, 3, 0, tzinfo=UTC)
        service = PricePredictionService(StubRepository([latest]), clock=lambda: now)

        result = service.predict_seven_days([3])

        self.assertEqual(7, len(result))
        self.assertEqual(
            [date(2026, 9, day) for day in range(19, 26)],
            [prediction.target_date for prediction in result],
        )
        self.assertTrue(
            all(prediction.predicted_price == latest.base_price for prediction in result)
        )
        self.assertTrue(all(prediction.generated_at == now for prediction in result))


if __name__ == "__main__":
    unittest.main()
