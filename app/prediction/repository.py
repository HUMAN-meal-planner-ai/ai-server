from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any


# 기존 단일 series 기준 최신 실측 가격 조회
LATEST_PRICES_SQL = """
SELECT DISTINCT ON (ps.series_id)
    ps.series_id,
    ip.price_date AS base_date,
    ip.standard_unit_price AS base_price,
    i.standard_unit
FROM mealfit.price_series ps
JOIN mealfit.ingredient i
  ON i.ingredient_id = ps.ingredient_id
JOIN mealfit.ingredient_price ip
  ON ip.series_id = ps.series_id
WHERE ps.series_id = ANY(%s)
ORDER BY
    ps.series_id,
    ip.price_date DESC,
    ip.price_id DESC
"""


@dataclass(frozen=True)
class LatestSeriesPrice:
    """series별 최신 실측 가격."""

    series_id: int
    base_date: date
    base_price: Decimal
    standard_unit: str


class PricePredictionRepository:
    """
    기존 가격예측 기능에서 사용하는 최신 실측 가격을 조회한다.

    주간 Ridge 예측용 가격 이력은 Backend가 조회하여
    AI Server 요청 데이터로 전달하도록 분리한다.
    """

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def find_latest_prices(
        self,
        series_ids: list[int],
    ) -> list[LatestSeriesPrice]:
        """요청한 series별 최신 실측 가격을 조회한다."""

        if not series_ids:
            return []

        with self._connection.cursor() as cursor:
            cursor.execute(
                LATEST_PRICES_SQL,
                (series_ids,),
            )

            return [
                LatestSeriesPrice(*row)
                for row in cursor.fetchall()
            ]