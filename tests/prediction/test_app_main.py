from __future__ import annotations

import unittest

from app.main import app


class AppMainTest(unittest.TestCase):
    def test_registers_price_prediction_router(self) -> None:
        paths = app.openapi()["paths"]

        self.assertIn("/api/v1/price-predictions/next", paths)
        self.assertIn("/api/v1/price-predictions/7-days", paths)


if __name__ == "__main__":
    unittest.main()
