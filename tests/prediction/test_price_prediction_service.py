from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from app.prediction.model_bundle import WeeklyPrediction
from app.prediction.repository import LatestSeriesPrice
from app.prediction.service import PricePredictionService, PriceSeriesNotFoundError
from app.prediction.weekly_service import WeeklyPricePredictionService
from app.schemas.price_prediction import (
    WeeklyPriceHistoryPoint,
    WeeklyPricePredictionBatchRequest,
    WeeklyPriceSeriesRequest,
)


class StubRepository:
    def __init__(self, prices: list[LatestSeriesPrice]) -> None:
        self.prices = prices

    def find_latest_prices(self, series_ids: list[int]) -> list[LatestSeriesPrice]:
        return self.prices


class StubWeeklyModelBundle:
    supported_series_ids = {3}
    model_name = "weekly_mean_ridge"
    model_version = "weekly_ridge_v1"

    def predict(self, feature: object) -> list[WeeklyPrediction]:
        return [
            WeeklyPrediction(
                predicted_mean_price=1.0,
                predicted_max_price=1.2,
                ridge_score=0.4,
                volatility_score=0.5,
                combined_risk_score=0.45,
            )
        ]


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


class WeeklyPricePredictionServiceTest(unittest.TestCase):
    def test_preserves_exact_base_price_from_request_history(self) -> None:
        base_date = date(2026, 9, 18)
        exact_base_price = Decimal("0.77466666666666666667")
        prices = [
            WeeklyPriceHistoryPoint(
                price_date=base_date - timedelta(days=28 - offset),
                representative_price=(
                    exact_base_price if offset == 28 else Decimal("0.800000")
                ),
            )
            for offset in range(29)
        ]
        request = WeeklyPricePredictionBatchRequest(
            series=[
                WeeklyPriceSeriesRequest(
                    series_id=3,
                    ingredient_code="F00993",
                    standard_unit="g",
                    prices=prices,
                )
            ]
        )
        service = WeeklyPricePredictionService(StubWeeklyModelBundle())

        result = service.predict(request)[0]

        self.assertEqual(exact_base_price, result.base_price)
        self.assertEqual(base_date, result.base_date)
        self.assertEqual(base_date + timedelta(days=7), result.target_date)


if __name__ == "__main__":
    unittest.main()
