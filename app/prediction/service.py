from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from app.prediction.repository import LatestSeriesPrice
from app.schemas.price_prediction import PricePredictionResponse


MODEL_NAME = "lag_1_baseline"
MODEL_VERSION = "lag_1_baseline_v1"
SEVEN_DAY_HORIZON = 7


class LatestPriceRepository(Protocol):
    def find_latest_prices(self, series_ids: list[int]) -> list[LatestSeriesPrice]: ...


class PriceSeriesNotFoundError(ValueError):
    """요청한 series에 실측 가격이 없을 때 발생한다."""


class PricePredictionService:
    def __init__(
        self,
        repository: LatestPriceRepository,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock or (lambda: datetime.now(UTC))

    def predict_next(self, series_ids: Sequence[int]) -> list[PricePredictionResponse]:
        return self._predict_horizon(series_ids, horizon_days=1)

    def predict_seven_days(
        self, series_ids: Sequence[int]
    ) -> list[PricePredictionResponse]:
        return self._predict_horizon(series_ids, horizon_days=SEVEN_DAY_HORIZON)

    def _predict_horizon(
        self, series_ids: Sequence[int], horizon_days: int
    ) -> list[PricePredictionResponse]:
        requested_ids = list(series_ids)
        latest_by_id = {
            price.series_id: price
            for price in self._repository.find_latest_prices(requested_ids)
        }
        missing_ids = [series_id for series_id in requested_ids if series_id not in latest_by_id]
        if missing_ids:
            joined_ids = ", ".join(str(series_id) for series_id in missing_ids)
            raise PriceSeriesNotFoundError(
                "최신 실측 가격을 찾을 수 없는 시계열이 있습니다: " + joined_ids
            )

        generated_at = self._clock()
        return [
            self._predict(latest_by_id[series_id], generated_at, days_ahead)
            for series_id in requested_ids
            for days_ahead in range(1, horizon_days + 1)
        ]

    @staticmethod
    def _predict(
        latest: LatestSeriesPrice, generated_at: datetime, days_ahead: int
    ) -> PricePredictionResponse:
        return PricePredictionResponse(
            series_id=latest.series_id,
            base_date=latest.base_date,
            target_date=latest.base_date + timedelta(days=days_ahead),
            base_price=latest.base_price,
            predicted_price=latest.base_price,
            standard_unit=latest.standard_unit,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            generated_at=generated_at,
        )
