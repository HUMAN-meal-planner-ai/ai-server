from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual, predicted) -> dict[str, float | int | None]:
    """0 target와 상수 target을 숨기지 않고 회귀 평가 지표를 계산한다."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape or actual_values.size == 0:
        raise ValueError("평가할 실제값과 예측값의 크기가 같고 비어 있지 않아야 합니다.")

    non_zero = actual_values != 0
    mape = None
    if non_zero.any():
        mape = float(
            np.mean(
                np.abs(
                    (actual_values[non_zero] - predicted_values[non_zero])
                    / actual_values[non_zero]
                )
            )
            * 100
        )

    r_squared = None
    if actual_values.size >= 2 and not np.allclose(actual_values, actual_values[0]):
        r_squared = float(r2_score(actual_values, predicted_values))

    return {
        "row_count": int(actual_values.size),
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(math.sqrt(mean_squared_error(actual_values, predicted_values))),
        "mape_percent": mape,
        "mape_excluded_zero_count": int((~non_zero).sum()),
        "r2": r_squared,
    }


def calculate_series_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, series_id), group in predictions.groupby(
        ["model", "series_id"], sort=True
    ):
        metrics = calculate_metrics(group["actual"], group["predicted"])
        rows.append({"model": model_name, "series_id": series_id, **metrics})
    return pd.DataFrame(rows)
