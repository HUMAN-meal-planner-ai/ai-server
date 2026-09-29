from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.price_prediction import (
    get_price_prediction_service,
    get_weekly_price_prediction_service,
    router,
)
from app.prediction.repository import LatestSeriesPrice
from app.prediction.service import PricePredictionService
from app.schemas.price_prediction import WeeklyPricePredictionResponse


FIXED_TIME = datetime(2026, 9, 22, 3, 0, tzinfo=UTC)


class FakeRepository:
    def __init__(self, prices: list[LatestSeriesPrice]) -> None:
        self._prices = prices

    def find_latest_prices(self, series_ids: list[int]) -> list[LatestSeriesPrice]:
        return [price for price in self._prices if price.series_id in series_ids]


class FakeWeeklyPricePredictionService:
    def predict(self, request):
        return [
            WeeklyPricePredictionResponse(
                series_id=series.series_id,
                base_date=series.prices[-1].price_date,
                target_date=series.prices[-1].price_date + timedelta(days=7),
                base_price=series.prices[-1].representative_price,
                predicted_price=series.prices[-1].representative_price,
                predicted_max_price=series.prices[-1].representative_price,
                standard_unit=series.standard_unit,
                ridge_score=0.5,
                volatility_score=0.4,
                combined_risk_score=0.45,
                model_name="weekly_mean_ridge",
                model_version="weekly_ridge_v1",
                generated_at=FIXED_TIME,
            )
            for series in request.series
        ]


class PricePredictionApiTest(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(router)
        repository = FakeRepository(
            [
                LatestSeriesPrice(3, date(2026, 9, 18), Decimal("0.820000"), "g"),
                LatestSeriesPrice(17, date(2026, 9, 17), Decimal("12.500000"), "g"),
            ]
        )
        service = PricePredictionService(repository, clock=lambda: FIXED_TIME)
        app.dependency_overrides[get_price_prediction_service] = lambda: service
        app.dependency_overrides[get_weekly_price_prediction_service] = (
            FakeWeeklyPricePredictionService
        )
        self.client = TestClient(app)

    def test_returns_lag_1_predictions_in_requested_order(self) -> None:
        response = self.client.post(
            "/api/v1/price-predictions/next", json={"seriesIds": [17, 3]}
        )

        self.assertEqual(200, response.status_code)
        body = response.json()["predictions"]
        self.assertEqual([17, 3], [item["seriesId"] for item in body])
        self.assertEqual("2026-09-17", body[0]["baseDate"])
        self.assertEqual("2026-09-18", body[0]["targetDate"])
        self.assertEqual("12.500000", body[0]["basePrice"])
        self.assertEqual(body[0]["basePrice"], body[0]["predictedPrice"])
        self.assertEqual("lag_1_baseline", body[0]["modelName"])
        self.assertEqual("lag_1_baseline_v1", body[0]["modelVersion"])
        self.assertEqual("g", body[0]["standardUnit"])

    def test_returns_not_found_when_a_series_has_no_actual_price(self) -> None:
        response = self.client.post(
            "/api/v1/price-predictions/next", json={"seriesIds": [3, 999]}
        )

        self.assertEqual(404, response.status_code)
        self.assertEqual(
            "최신 실측 가격을 찾을 수 없는 시계열이 있습니다: 999",
            response.json()["detail"],
        )

    def test_rejects_duplicate_series_ids(self) -> None:
        response = self.client.post(
            "/api/v1/price-predictions/next", json={"seriesIds": [3, 3]}
        )

        self.assertEqual(422, response.status_code)
        self.assertIn(
            "시계열 ID는 중복해서 요청할 수 없습니다.",
            response.json()["detail"][0]["msg"],
        )

    def test_returns_seven_day_predictions_for_each_series(self) -> None:
        response = self.client.post(
            "/api/v1/price-predictions/7-days",
            json={
                "series": [
                    {
                        "seriesId": 3,
                        "ingredientCode": "ING-003",
                        "standardUnit": "g",
                        "prices": [
                            {"priceDate": "2026-09-17", "representativePrice": "0.80"},
                            {"priceDate": "2026-09-18", "representativePrice": "0.82"},
                        ],
                    },
                    {
                        "seriesId": 17,
                        "ingredientCode": "ING-017",
                        "standardUnit": "g",
                        "prices": [
                            {"priceDate": "2026-09-16", "representativePrice": "12.40"},
                            {"priceDate": "2026-09-17", "representativePrice": "12.50"},
                        ],
                    },
                ]
            },
        )

        self.assertEqual(200, response.status_code)
        body = response.json()["predictions"]
        self.assertEqual(2, len(body))
        self.assertEqual([3, 17], [item["seriesId"] for item in body])
        self.assertEqual(["2026-09-25", "2026-09-24"], [item["targetDate"] for item in body])
        self.assertEqual(["0.82", "12.50"], [item["basePrice"] for item in body])
        self.assertTrue(all(item["modelVersion"] == "weekly_ridge_v1" for item in body))


if __name__ == "__main__":
    unittest.main()
