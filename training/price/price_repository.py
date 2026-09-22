from __future__ import annotations

from typing import Any

import pandas as pd


SERIES_ELIGIBILITY_SQL = """
SELECT
    ps.series_id,
    ps.ingredient_id,
    i.ingredient_code,
    i.name AS ingredient_name,
    count(ip.price_id) AS observation_count,
    min(ip.price_date) AS first_price_date,
    max(ip.price_date) AS last_price_date
FROM mealfit.price_series ps
JOIN mealfit.ingredient i ON i.ingredient_id = ps.ingredient_id
LEFT JOIN mealfit.ingredient_price ip ON ip.series_id = ps.series_id
WHERE ps.source_name = 'KAMIS'
GROUP BY ps.series_id, ps.ingredient_id, i.ingredient_code, i.name
ORDER BY ps.series_id
"""


ELIGIBLE_PRICES_SQL = """
WITH eligible_series AS (
    SELECT ps.series_id
    FROM mealfit.price_series ps
    JOIN mealfit.ingredient_price ip ON ip.series_id = ps.series_id
    WHERE ps.source_name = 'KAMIS'
    GROUP BY ps.series_id
    HAVING count(ip.price_id) >= %s
)
SELECT
    ps.series_id,
    ps.ingredient_id,
    i.ingredient_code,
    i.name AS ingredient_name,
    ps.source_category_code,
    ps.source_item_code,
    ps.source_kind_code,
    ps.source_rank_code,
    ps.variety,
    ps.grade,
    ps.price_type,
    ps.market,
    ps.region,
    ps.original_unit,
    ps.unit_quantity,
    i.standard_unit,
    ip.price_date,
    ip.original_price,
    ip.standard_unit_price
FROM mealfit.price_series ps
JOIN eligible_series eligible ON eligible.series_id = ps.series_id
JOIN mealfit.ingredient i ON i.ingredient_id = ps.ingredient_id
JOIN mealfit.ingredient_price ip ON ip.series_id = ps.series_id
ORDER BY ps.series_id, ip.price_date
"""


class PriceDataRepository:
    """가격 feature 생성에 필요한 원천 데이터를 읽기 전용으로 조회한다."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def load_series_eligibility(self) -> pd.DataFrame:
        return self._query(SERIES_ELIGIBILITY_SQL)

    def load_eligible_prices(self, minimum_observations: int = 200) -> pd.DataFrame:
        if minimum_observations < 1:
            raise ValueError("최소 관측 건수는 1건 이상이어야 합니다.")
        return self._query(ELIGIBLE_PRICES_SQL, (minimum_observations,))

    def _query(self, sql: str, parameters: tuple[Any, ...] | None = None) -> pd.DataFrame:
        with self._connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
        return pd.DataFrame(rows, columns=columns)
