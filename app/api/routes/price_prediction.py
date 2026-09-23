from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.prediction.repository import (
    PricePredictionRepository,
)
from app.prediction.service import (
    PricePredictionService,
    PriceSeriesNotFoundError,
)
from app.prediction.weekly_service import (
    UnsupportedPredictionSeriesError,
    WeeklyPricePredictionService,
)
from app.prediction.feature_builder import (
    InsufficientPriceHistoryError,
)
from app.schemas.price_prediction import (
    PricePredictionBatchRequest,
    PricePredictionBatchResponse,
    WeeklyPricePredictionBatchRequest,
    WeeklyPricePredictionBatchResponse,
)
from training.price.config import DatabaseSettings
from training.price.database import (
    open_read_only_connection,
)


router = APIRouter(
    prefix="/api/v1/price-predictions",
    tags=["price-prediction"],
)


# ------------------------------------------------------------------
# 기존 /next 예측 서비스
# ------------------------------------------------------------------

def get_price_prediction_service(
) -> Iterator[PricePredictionService]:
    """
    기존 단기 예측용 서비스 의존성을 생성한다.

    기존 /next API는 AI Server가 DB에서 최신 실측 가격을 직접
    조회하는 구조이므로 PostgreSQL 연결을 사용한다.

    이 구조는 기존 기능 호환을 위해 현재 유지한다.
    신규 주간 Ridge 예측은 DB에 직접 접근하지 않는다.
    """
    settings = DatabaseSettings.from_environment()

    with open_read_only_connection(
        settings
    ) as connection:
        yield PricePredictionService(
            PricePredictionRepository(connection)
        )


# ------------------------------------------------------------------
# 신규 /7-days Ridge 예측 서비스
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_weekly_price_prediction_service(
) -> WeeklyPricePredictionService:
    """
    주간 Ridge 예측 서비스를 애플리케이션 프로세스당 한 번 생성한다.

    WeeklyPricePredictionService를 요청마다 새로 만들면
    preprocessor.joblib, Ridge 모델, calibration artifact를
    매 요청마다 디스크에서 다시 읽게 된다.

    따라서 lru_cache를 사용해 최초 요청 시 한 번만 로드하고
    이후 요청에서는 같은 모델 객체를 재사용한다.

    주간 예측 서비스는 DB에 직접 연결하지 않는다.
    가격 이력은 Backend가 요청 body로 전달한다.
    """
    return WeeklyPricePredictionService()


# ------------------------------------------------------------------
# 기존 다음 가격 예측
# ------------------------------------------------------------------

@router.post(
    "/next",
    response_model=PricePredictionBatchResponse,
)
def predict_next_prices(
    request: PricePredictionBatchRequest,
    service: PricePredictionService = Depends(
        get_price_prediction_service
    ),
) -> PricePredictionBatchResponse:
    """
    기존 lag_1 기반 단기 가격예측 API.

    기존 기능 호환을 위해 현재 구현을 유지한다.
    """
    try:
        predictions = service.predict_next(
            request.series_ids
        )

    except PriceSeriesNotFoundError as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exception),
        ) from exception

    return PricePredictionBatchResponse(
        predictions=predictions
    )


# ------------------------------------------------------------------
# 신규 다음 7일 Ridge 예측
# ------------------------------------------------------------------

@router.post(
    "/7-days",
    response_model=WeeklyPricePredictionBatchResponse,
)
def predict_seven_day_prices(
    request: WeeklyPricePredictionBatchRequest,
    service: WeeklyPricePredictionService = Depends(
        get_weekly_price_prediction_service
    ),
) -> WeeklyPricePredictionBatchResponse:
    """
    Backend가 전달한 3지역 대표가격 이력을 사용해
    다음 7일 가격을 Ridge 모델로 예측한다.

    반환값:
    - 다음 7일 평균 예상가격
    - 다음 7일 최대 예상가격
    - Ridge percentile 점수
    - 최근 변동성 percentile 점수
    - ML 기반 combined risk score

    최종 위험등급 판정은 Backend에서 처리한다.
    """
    try:
        predictions = service.predict(request)

    # 현재 운영 artifact가 학습하지 않은 series가 들어온 경우
    except UnsupportedPredictionSeriesError as exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exception),
        ) from exception

    # 28일 이력 등 feature 생성 조건이 충족되지 않은 경우
    except InsufficientPriceHistoryError as exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exception),
        ) from exception

    return WeeklyPricePredictionBatchResponse(
        predictions=predictions
    )