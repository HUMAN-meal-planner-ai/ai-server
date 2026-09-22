from __future__ import annotations

import unittest

import pandas as pd

from training.price.feature_engineering import FEATURE_COLUMNS
from training.price.model_definitions import build_preprocessor


class ModelDefinitionsTest(unittest.TestCase):
    def test_series_id_is_train_fitted_as_categorical_one_hot(self) -> None:
        frame = pd.DataFrame(
            {
                **{
                    column: [1.0, 2.0]
                    for column in FEATURE_COLUMNS
                    if column != "season"
                },
                "season": ["WINTER", "SPRING"],
                "series_id": [3, 8],
            }
        )
        preprocessor = build_preprocessor()

        transformed = preprocessor.fit_transform(frame)
        categories = preprocessor.named_transformers_["categorical"].categories_

        self.assertEqual([3, 8], categories[1].tolist())
        self.assertGreater(transformed.shape[1], len(FEATURE_COLUMNS))
