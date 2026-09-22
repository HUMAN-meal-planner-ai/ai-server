from __future__ import annotations

from collections.abc import Iterator
from fastapi import APIRouter, Depends, HTTPException, status

from app.prediction.repository import PricePredictionRepository
from app.prediction.service import PricePredictionService, PriceSeriesNotFoundError
from app.schemas.price_prediction import (
    PricePredictionBatchRequest,
    PricePredictionBatchResponse,
)
from training.price.config import DatabaseSettings
from training.price.database import open_read_only_connection


router = APIRouter(prefix="/api/v1/price-predictions", tags=["price-prediction"])


def get_price_prediction_service() -> Iterator[PricePredictionService]:
    settings = DatabaseSettings.from_environment()
    with open_read_only_connection(settings) as connection:
        yield PricePredictionService(PricePredictionRepository(connection))


@router.post("/next", response_model=PricePredictionBatchResponse)
def predict_next_prices(
    request: PricePredictionBatchRequest,
    service: PricePredictionService = Depends(get_price_prediction_service),
) -> PricePredictionBatchResponse:
    try:
        predictions = service.predict_next(request.series_ids)
    except PriceSeriesNotFoundError as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exception),
        ) from exception
    return PricePredictionBatchResponse(predictions=predictions)


@router.post("/7-days", response_model=PricePredictionBatchResponse)
def predict_seven_day_prices(
    request: PricePredictionBatchRequest,
    service: PricePredictionService = Depends(get_price_prediction_service),
) -> PricePredictionBatchResponse:
    try:
        predictions = service.predict_seven_days(request.series_ids)
    except PriceSeriesNotFoundError as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exception),
        ) from exception
    return PricePredictionBatchResponse(predictions=predictions)
