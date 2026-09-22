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


if __name__ == "__main__":
    unittest.main()
