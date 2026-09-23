from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
)


def _to_camel_case(name: str) -> str:
    """Python snake_case 필드명을 API camelCase로 변환한다."""
    words = name.split("_")
    return words[0] + "".join(
        word.capitalize()
        for word in words[1:]
    )


class ApiSchema(BaseModel):
    """밀핏 AI Server API 공통 스키마."""

    model_config = ConfigDict(
        alias_generator=_to_camel_case,
        populate_by_name=True,
    )


# ------------------------------------------------------------------
# 기존 단일 가격예측 API
# ------------------------------------------------------------------

class PricePredictionBatchRequest(ApiSchema):
    """기존 /next 가격예측 요청."""

    series_ids: list[PositiveInt] = Field(
        min_length=1,
        max_length=200,
    )

    @field_validator("series_ids")
    @classmethod
    def validate_unique_series_ids(
        cls,
        value: list[int],
    ) -> list[int]:
        """동일한 series를 한 요청에서 중복 요청하지 못하게 한다."""
        if len(value) != len(set(value)):
            raise ValueError(
                "시계열 ID는 중복해서 요청할 수 없습니다."
            )

        return value


class PricePredictionResponse(ApiSchema):
    """기존 가격예측 응답."""

    series_id: int
    base_date: date
    target_date: date
    base_price: Decimal
    predicted_price: Decimal
    standard_unit: str
    model_name: str
    model_version: str
    generated_at: datetime


class PricePredictionBatchResponse(ApiSchema):
    """기존 가격예측 일괄 응답."""

    predictions: list[PricePredictionResponse]


# ------------------------------------------------------------------
# 주간 Ridge 가격예측 API
# ------------------------------------------------------------------

class WeeklyPriceHistoryPoint(ApiSchema):
    """
    Backend가 AI Server에 전달하는 일별 3지역 대표가격.

    representative_price는 같은 날짜의
    서울·부산·대전 표준단가 평균값이다.
    """

    price_date: date

    representative_price: Decimal = Field(
        gt=0,
    )


class WeeklyPriceSeriesRequest(ApiSchema):
    """
    하나의 canonical series에 대한 주간예측 입력 데이터.

    series_id는 학습에 사용한 서울 기준 canonical series ID이며,
    prices에는 3지역 대표가격 이력이 들어간다.
    """

    series_id: PositiveInt
    ingredient_code: str = Field(
        min_length=1,
        max_length=100,
    )
    standard_unit: str = Field(
        min_length=1,
        max_length=30,
    )

    prices: list[WeeklyPriceHistoryPoint] = Field(
        min_length=2,
        max_length=400,
    )

    @field_validator("prices")
    @classmethod
    def validate_unique_price_dates(
        cls,
        value: list[WeeklyPriceHistoryPoint],
    ) -> list[WeeklyPriceHistoryPoint]:
        """
        하나의 series에서 동일 날짜가 중복 전달되는 것을 방지한다.

        28일 이력 충족 여부는 실제 feature 생성 단계에서 검증한다.
        """
        dates = [
            point.price_date
            for point in value
        ]

        if len(dates) != len(set(dates)):
            raise ValueError(
                "대표가격 이력에 중복된 날짜가 있습니다."
            )

        return value


class WeeklyPricePredictionBatchRequest(ApiSchema):
    """
    Backend → AI Server 주간 가격예측 요청.

    DB 조회와 3지역 평균 계산은 Backend 책임이고,
    AI Server는 전달받은 이력으로 ML feature를 생성한다.
    """

    series: list[WeeklyPriceSeriesRequest] = Field(
        min_length=1,
        max_length=200,
    )

    @field_validator("series")
    @classmethod
    def validate_unique_series_ids(
        cls,
        value: list[WeeklyPriceSeriesRequest],
    ) -> list[WeeklyPriceSeriesRequest]:
        """동일 canonical series의 중복 요청을 방지한다."""
        series_ids = [
            item.series_id
            for item in value
        ]

        if len(series_ids) != len(set(series_ids)):
            raise ValueError(
                "시계열 ID는 중복해서 요청할 수 없습니다."
            )

        return value


class WeeklyPricePredictionResponse(ApiSchema):
    """
    주간 Ridge 모델의 예측 결과.

    predicted_price:
        다음 7일 평균 예상가격.
        Backend의 price_prediction 테이블에는 이 값을 저장한다.

    predicted_max_price:
        다음 7일 최대 예상가격.
        위험도 계산용 신호이며 현재 DB에는 저장하지 않는다.

    ridge_score / volatility_score / combined_risk_score:
        ML에서 계산한 0~1 위험 점수.
        최종 위험등급 판단은 Backend에서 처리한다.
    """

    series_id: int
    base_date: date
    target_date: date

    # 현재 3지역 대표가격
    base_price: Decimal

    # 다음 7일 평균 예상가격
    predicted_price: Decimal

    # 다음 7일 최대 예상가격
    predicted_max_price: Decimal

    standard_unit: str

    ridge_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    volatility_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    combined_risk_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    model_name: str
    model_version: str
    generated_at: datetime


class WeeklyPricePredictionBatchResponse(ApiSchema):
    """주간 Ridge 가격예측 일괄 응답."""

    predictions: list[WeeklyPricePredictionResponse]