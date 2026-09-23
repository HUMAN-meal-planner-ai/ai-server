from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplitBoundaries:
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def determine_time_boundaries(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TimeSplitBoundaries:
    """모든 series에 동일하게 적용할 전역 날짜 경계를 계산한다."""
    if "price_date" not in frame.columns:
        raise ValueError("시간순 분할에 price_date 컬럼이 필요합니다.")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("학습 및 검증 데이터 비율은 0보다 커야 합니다.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("학습 및 검증 데이터 비율의 합은 1보다 작아야 합니다.")

    dates = pd.DatetimeIndex(pd.to_datetime(frame["price_date"]).drop_duplicates().sort_values())
    train_size = int(len(dates) * train_fraction)
    validation_end_index = int(len(dates) * (train_fraction + validation_fraction))
    if train_size < 1 or validation_end_index <= train_size or validation_end_index >= len(dates):
        raise ValueError("시간순 학습·검증·테스트 분할에 필요한 날짜가 부족합니다.")

    return TimeSplitBoundaries(
        train_end=dates[train_size - 1],
        validation_end=dates[validation_end_index - 1],
    )


def split_name_for_dates(
    dates: pd.Series, boundaries: TimeSplitBoundaries
) -> pd.Series:
    normalized = pd.to_datetime(dates)
    result = pd.Series("test", index=dates.index, dtype="string")
    result.loc[normalized <= boundaries.validation_end] = "validation"
    result.loc[normalized <= boundaries.train_end] = "train"
    return result
