from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from training.price.config import _normalize_postgresql_url
from training.price.price_repository import PriceDataRepository


class PriceDataRepositoryTest(unittest.TestCase):
    def test_jdbc_url_is_normalized_for_psycopg(self) -> None:
        self.assertEqual(
            "postgresql://localhost:5432/mealfit",
            _normalize_postgresql_url("jdbc:postgresql://localhost:5432/mealfit"),
        )

    def test_eligible_price_query_uses_minimum_count_and_order(self) -> None:
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.description = [
            SimpleNamespace(name="series_id"),
            SimpleNamespace(name="price_date"),
        ]
        cursor.fetchall.return_value = [(3, "2026-09-18")]
        connection = MagicMock()
        connection.cursor.return_value = cursor

        result = PriceDataRepository(connection).load_eligible_prices(200)

        executed_sql, parameters = cursor.execute.call_args.args
        self.assertIn("HAVING count(ip.price_id) >= %s", executed_sql)
        self.assertIn("ORDER BY ps.series_id, ip.price_date", executed_sql)
        self.assertEqual((200,), parameters)
        self.assertEqual([3], result["series_id"].tolist())

    def test_non_positive_minimum_observation_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "1건 이상"):
            PriceDataRepository(MagicMock()).load_eligible_prices(0)

    def test_regional_representative_price_query_requires_all_three_regions(self) -> None:
        map_cursor = MagicMock()
        map_cursor.__enter__.return_value = map_cursor
        map_cursor.description = [
            SimpleNamespace(name=name)
            for name in ("series_id", "regional_series_id", "region", "ingredient_code")
        ]
        map_cursor.fetchall.return_value = [
            (3, 3, "서울", "F00993"),
            (3, 62, "부산", "F00993"),
            (3, 63, "대전", "F00993"),
        ]
        price_cursor = MagicMock()
        price_cursor.__enter__.return_value = price_cursor
        price_cursor.description = [
            SimpleNamespace(name=name)
            for name in (
                "series_id",
                "price_date",
                "representative_standard_unit_price",
            )
        ]
        price_cursor.fetchall.return_value = [(3, "2026-09-15", 2.5)]
        connection = MagicMock()
        connection.cursor.side_effect = [map_cursor, price_cursor]

        result = PriceDataRepository(connection).load_regional_representative_prices(200)

        map_sql, map_parameters = map_cursor.execute.call_args.args
        price_sql, price_parameters = price_cursor.execute.call_args.args
        self.assertIn("regional.region IN", map_sql)
        self.assertIn("AVG(price.standard_unit_price)", price_sql)
        self.assertEqual(([4, 50, 59],), map_parameters)
        self.assertEqual(([3, 3, 3], [3, 62, 63], 200), price_parameters)
        self.assertEqual([3], result["series_id"].tolist())


if __name__ == "__main__":
    unittest.main()
