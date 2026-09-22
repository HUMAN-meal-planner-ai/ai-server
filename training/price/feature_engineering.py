from __future__ import annotations

import numpy as np
import pandas as pd


LAG_PERIODS = (1, 7, 14, 30)
ROLLING_MEAN_WINDOWS = (7, 14, 30)
FEATURE_COLUMNS = (
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "pct_change_1",
    "pct_change_7",
    "days_since_previous",
    "day_of_week",
    "month",
    "season",
)
TARGET_COLUMN = "target_standard_unit_price"

REQUIRED_COLUMNS = {
    "series_id",
    "price_date",
    "original_price",
    "standard_unit_price",
}


def build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """각 series의 과거 관측값만 사용해 누수 없는 가격 특성을 생성한다."""
    missing_columns = sorted(REQUIRED_COLUMNS.difference(prices.columns))
    if missing_columns:
        raise ValueError("가격 원천 데이터의 필수 컬럼이 없습니다: " + ", ".join(missing_columns))
    if prices.empty:
        raise ValueError("가격 특성을 생성할 원천 데이터가 없습니다.")

    frame = prices.copy()
    frame["price_date"] = pd.to_datetime(frame["price_date"], errors="raise")
    frame["original_price"] = pd.to_numeric(frame["original_price"], errors="raise")
    frame["standard_unit_price"] = pd.to_numeric(
        frame["standard_unit_price"], errors="raise"
    )
    frame = frame.sort_values(["series_id", "price_date"], kind="stable").reset_index(drop=True)

    duplicate_mask = frame.duplicated(["series_id", "price_date"], keep=False)
    if duplicate_mask.any():
        raise ValueError("같은 시계열과 날짜의 가격 행이 중복되어 있습니다.")
    if frame["standard_unit_price"].isna().any():
        raise ValueError("표준 단위 가격이 비어 있는 행이 있습니다.")
    if (frame["standard_unit_price"] <= 0).any():
        raise ValueError("표준 단위 가격은 0보다 커야 합니다.")

    grouped_price = frame.groupby("series_id", sort=False)["standard_unit_price"]
    for period in LAG_PERIODS:
        frame[f"lag_{period}"] = grouped_price.shift(period)

    # 현재 가격을 rolling 계산에서 제외해 target 값이 feature로 새지 않게 한다.
    previous_prices = grouped_price.shift(1)
    previous_grouped = previous_prices.groupby(frame["series_id"], sort=False)
    for window in ROLLING_MEAN_WINDOWS:
        frame[f"rolling_mean_{window}"] = previous_grouped.transform(
            lambda values, size=window: values.rolling(size, min_periods=size).mean()
        )
    frame["rolling_std_7"] = previous_grouped.transform(
        lambda values: values.rolling(7, min_periods=7).std(ddof=0)
    )

    lag_2 = grouped_price.shift(2)
    lag_8 = grouped_price.shift(8)
    frame["pct_change_1"] = frame["lag_1"].div(lag_2).sub(1)
    frame["pct_change_7"] = frame["lag_1"].div(lag_8).sub(1)
    frame[["pct_change_1", "pct_change_7"]] = frame[
        ["pct_change_1", "pct_change_7"]
    ].replace([np.inf, -np.inf], np.nan)

    frame["days_since_previous"] = (
        frame.groupby("series_id", sort=False)["price_date"].diff().dt.days
    )
    frame["day_of_week"] = frame["price_date"].dt.dayofweek
    frame["month"] = frame["price_date"].dt.month
    frame["season"] = frame["month"].map(_season_for_month)
    frame[TARGET_COLUMN] = frame.pop("standard_unit_price")

    # 최대 30개 과거 관측이 모두 존재하는 행만 남기며 미래 값으로 결측을 채우지 않는다.
    frame = frame.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)
    return frame


def _season_for_month(month: int) -> str:
    if month in (3, 4, 5):
        return "SPRING"
    if month in (6, 7, 8):
        return "SUMMER"
    if month in (9, 10, 11):
        return "AUTUMN"
    return "WINTER"
