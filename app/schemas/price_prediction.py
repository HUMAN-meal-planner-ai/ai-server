from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


def _to_camel_case(name: str) -> str:
    words = name.split("_")
    return words[0] + "".join(word.capitalize() for word in words[1:])


class ApiSchema(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel_case, populate_by_name=True)


class PricePredictionBatchRequest(ApiSchema):
    series_ids: list[PositiveInt] = Field(min_length=1, max_length=200)

    @field_validator("series_ids")
    @classmethod
    def validate_unique_series_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("시계열 ID는 중복해서 요청할 수 없습니다.")
        return value


class PricePredictionResponse(ApiSchema):
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
    predictions: list[PricePredictionResponse]
