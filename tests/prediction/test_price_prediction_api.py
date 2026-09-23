from __future__ import annotations

import unittest
from datetime import UTC, date, datetime
from decimal import Decimal

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.price_prediction import get_price_prediction_service, router
from app.prediction.repository import LatestSeriesPrice
from app.prediction.service import PricePredictionService


FIXED_TIME = datetime(2026, 9, 22, 3, 0, tzinfo=UTC)


class FakeRepository:
    def __init__(self, prices: list[LatestSeriesPrice]) -> None:
        self._prices = prices

    def find_latest_prices(self, series_ids: list[int]) -> list[LatestSeriesPrice]:
        return [price for price in self._prices if price.series_id in series_ids]


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
            "/api/v1/price-predictions/7-days", json={"seriesIds": [3, 17]}
        )

        self.assertEqual(200, response.status_code)
        body = response.json()["predictions"]
        self.assertEqual(14, len(body))
        series_three = [item for item in body if item["seriesId"] == 3]
        self.assertEqual(
            [f"2026-09-{day:02d}" for day in range(19, 26)],
            [item["targetDate"] for item in series_three],
        )
        self.assertTrue(
            all(item["predictedPrice"] == "0.820000" for item in series_three)
        )
        self.assertTrue(
            all(item["modelName"] == "lag_1_baseline" for item in series_three)
        )


if __name__ == "__main__":
    unittest.main()
