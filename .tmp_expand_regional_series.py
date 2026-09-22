from __future__ import annotations

import json
from pathlib import Path

import psycopg

from training.price.config import DatabaseSettings


ELIGIBLE_SQL = """
SELECT
    ps.series_id,
    ps.ingredient_id,
    i.ingredient_code,
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
    count(ip.price_id) AS observation_count
FROM mealfit.price_series ps
JOIN mealfit.ingredient i ON i.ingredient_id = ps.ingredient_id
JOIN mealfit.ingredient_price ip ON ip.series_id = ps.series_id
WHERE ps.source_name = 'KAMIS'
  AND ps.region = '서울'
  AND i.is_active = true
GROUP BY ps.series_id, ps.ingredient_id, i.ingredient_code
HAVING count(ip.price_id) >= 200
ORDER BY ps.series_id
"""


INSERT_SQL = """
INSERT INTO mealfit.price_series (
    ingredient_id, source_name,
    source_category_code, source_item_code, source_kind_code, source_rank_code,
    variety, grade, price_type, market, region,
    original_unit, unit_quantity, is_cost_basis
)
SELECT
    source.ingredient_id, source.source_name,
    source.source_category_code, source.source_item_code,
    source.source_kind_code, source.source_rank_code,
    source.variety, source.grade, source.price_type,
    'KAMIS 지역 도매 대표가격', target.region,
    source.original_unit, source.unit_quantity, false
FROM mealfit.price_series source
CROSS JOIN (VALUES ('부산'), ('대전')) AS target(region)
WHERE source.series_id = ANY(%s)
ON CONFLICT DO NOTHING
RETURNING series_id, ingredient_id, source_item_code,
          source_kind_code, source_rank_code, region
"""


def main() -> None:
    settings = DatabaseSettings.from_environment(Path("../backend/.env"))
    result = {}
    with psycopg.connect(
        settings.url,
        user=settings.username,
        password=settings.password,
    ) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(ELIGIBLE_SQL)
                columns = [column.name for column in cursor.description]
                eligible = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
                source_ids = [row["series_id"] for row in eligible]

                cursor.execute(INSERT_SQL, (source_ids,))
                inserted = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT count(*)
                    FROM mealfit.price_series regional
                    WHERE regional.region IN ('부산', '대전')
                      AND EXISTS (
                          SELECT 1
                          FROM mealfit.price_series source
                          WHERE source.series_id = ANY(%s)
                            AND source.ingredient_id = regional.ingredient_id
                            AND source.source_name = regional.source_name
                            AND source.source_category_code = regional.source_category_code
                            AND source.source_item_code = regional.source_item_code
                            AND source.source_kind_code = regional.source_kind_code
                            AND source.source_rank_code = regional.source_rank_code
                            AND source.price_type = regional.price_type
                            AND source.original_unit = regional.original_unit
                            AND source.unit_quantity = regional.unit_quantity
                      )
                    """,
                    (source_ids,),
                )
                prepared_count = cursor.fetchone()[0]

                result = {
                    "eligible_series_count": len(eligible),
                    "eligible_ingredient_count": len(
                        {row["ingredient_code"] for row in eligible}
                    ),
                    "eligible_series_ids": source_ids,
                    "ingredient_codes": sorted(
                        {row["ingredient_code"] for row in eligible}
                    ),
                    "inserted_count": len(inserted),
                    "inserted_series": [
                        {
                            "series_id": row[0],
                            "ingredient_id": row[1],
                            "item_code": row[2],
                            "kind_code": row[3],
                            "rank_code": row[4],
                            "region": row[5],
                        }
                        for row in inserted
                    ],
                    "prepared_regional_series_count": prepared_count,
                }

    Path(".tmp_expand_regional_series_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
