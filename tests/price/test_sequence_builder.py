from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from training.price.sequence_builder import build_sequences, sequence_eligible_indices


class SequenceBuilderTest(unittest.TestCase):
    def test_sequences_do_not_cross_series_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "series_id": [1, 1, 1, 2, 2, 2],
                "target": [10, 11, 12, 20, 21, 22],
            }
        )
        transformed = np.arange(12, dtype=float).reshape(6, 2)
        indices = sequence_eligible_indices(frame, sequence_length=3)

        sequences, targets = build_sequences(
            transformed, frame, indices, sequence_length=3, target_column="target"
        )

        self.assertEqual([2, 5], indices.tolist())
        self.assertEqual((2, 3, 2), sequences.shape)
        self.assertEqual([12.0, 22.0], targets.tolist())
        np.testing.assert_array_equal(sequences[1], transformed[3:6])
