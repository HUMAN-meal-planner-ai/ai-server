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


REGIONAL_SERIES_MAP_SQL = """
WITH source_series AS (
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
        ps.original_unit,
        ps.unit_quantity,
        i.standard_unit
    FROM mealfit.price_series ps
    JOIN mealfit.ingredient i ON i.ingredient_id = ps.ingredient_id
    WHERE ps.source_name = 'KAMIS'
      AND ps.region = '서울'
      AND ps.series_id <> ALL(%s)
)
SELECT
    source.*,
    regional.series_id AS regional_series_id,
    regional.region
FROM source_series source
JOIN mealfit.price_series regional
      ON regional.ingredient_id = source.ingredient_id
     AND regional.source_name = 'KAMIS'
     AND regional.source_category_code IS NOT DISTINCT FROM source.source_category_code
     AND regional.source_item_code IS NOT DISTINCT FROM source.source_item_code
     AND regional.source_kind_code IS NOT DISTINCT FROM source.source_kind_code
     AND regional.source_rank_code IS NOT DISTINCT FROM source.source_rank_code
     AND regional.price_type IS NOT DISTINCT FROM source.price_type
     AND regional.original_unit IS NOT DISTINCT FROM source.original_unit
     AND regional.unit_quantity IS NOT DISTINCT FROM source.unit_quantity
     AND regional.region IN ('서울', '부산', '대전')
ORDER BY source.series_id, regional.region
"""


REGIONAL_REPRESENTATIVE_PRICES_SQL = """
WITH regional_map AS (
    SELECT *
    FROM unnest(%s::bigint[], %s::bigint[])
        AS mapping(series_id, regional_series_id)
), daily_representative_prices AS (
    SELECT
        regional_map.series_id,
        price.price_date,
        AVG(price.standard_unit_price) AS representative_standard_unit_price
    FROM regional_map
    JOIN mealfit.ingredient_price price
      ON price.series_id = regional_map.regional_series_id
    GROUP BY regional_map.series_id, price.price_date
    HAVING count(*) = 3
       AND count(price.standard_unit_price) = 3
), eligible_series AS (
    SELECT series_id
    FROM daily_representative_prices
    GROUP BY series_id
    HAVING count(*) >= %s
)
SELECT representative.series_id,
       representative.price_date,
       representative.representative_standard_unit_price
FROM daily_representative_prices representative
JOIN eligible_series eligible ON eligible.series_id = representative.series_id
ORDER BY representative.series_id, representative.price_date
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

    def load_regional_representative_prices(
        self,
        minimum_observations: int = 200,
        excluded_series_ids: tuple[int, ...] = (4, 50, 59),
    ) -> pd.DataFrame:
        """서울·부산·대전이 모두 공시된 날짜만 Decimal AVG 대표가격으로 조회한다."""
        if minimum_observations < 1:
            raise ValueError("최소 관측 건수는 1건 이상이어야 합니다.")
        regional_map = self._query(
            REGIONAL_SERIES_MAP_SQL,
            (list(excluded_series_ids),),
        )
        regional_counts = regional_map.groupby("series_id")["region"].nunique()
        complete_series_ids = regional_counts[regional_counts.eq(3)].index.tolist()
        if not complete_series_ids:
            return pd.DataFrame()
        complete_map = regional_map[regional_map["series_id"].isin(complete_series_ids)]
        daily_prices = self._query(
            REGIONAL_REPRESENTATIVE_PRICES_SQL,
            (
                complete_map["series_id"].tolist(),
                complete_map["regional_series_id"].tolist(),
                minimum_observations,
            ),
        )
        metadata = complete_map.drop(columns=["regional_series_id", "region"]).drop_duplicates(
            "series_id"
        )
        return daily_prices.merge(metadata, on="series_id", how="inner", validate="many_to_one").sort_values(
            ["series_id", "price_date"], kind="stable"
        ).reset_index(drop=True)

    def _query(self, sql: str, parameters: tuple[Any, ...] | None = None) -> pd.DataFrame:
        with self._connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
        return pd.DataFrame(rows, columns=columns)
