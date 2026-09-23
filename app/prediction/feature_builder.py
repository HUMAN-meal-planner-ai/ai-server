from __future__ import annotations

import numpy as np
import pandas as pd

from training.price.compare_regional_weekly_models import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


class InsufficientPriceHistoryError(ValueError):
    """주간 예측 feature 생성에 필요한 가격 이력이 부족할 때 발생한다."""


def build_latest_weekly_feature(
    representative_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    3지역 일별 대표가격 이력에서 가장 최근 날짜 기준
    운영용 주간 예측 feature 1행을 생성한다.

    입력 필수 컬럼:
    - series_id
    - ingredient_code
    - price_date
    - representative_standard_unit_price

    학습 시 사용한 feature 계산 규칙을 그대로 사용하며
    미래 target 관련 컬럼은 생성하지 않는다.
    """
    required = {
        "series_id",
        "ingredient_code",
        "price_date",
        "representative_standard_unit_price",
    }

    missing = sorted(
        required.difference(representative_prices.columns)
    )
    if missing:
        raise ValueError(
            "운영 feature 생성에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    if representative_prices.empty:
        raise InsufficientPriceHistoryError(
            "대표가격 이력이 없습니다."
        )

    source = representative_prices.copy()

    # 날짜와 가격 타입을 학습 데이터와 동일하게 정규화한다.
    source["price_date"] = pd.to_datetime(
        source["price_date"],
        errors="raise",
    )

    source["representative_standard_unit_price"] = (
        pd.to_numeric(
            source["representative_standard_unit_price"],
            errors="raise",
        )
    )

    # 한 번에 하나의 대표 series만 feature로 만든다.
    if source["series_id"].nunique() != 1:
        raise ValueError(
            "feature 생성 요청에는 하나의 series_id만 포함되어야 합니다."
        )

    if source["ingredient_code"].nunique() != 1:
        raise ValueError(
            "feature 생성 요청에는 하나의 ingredient_code만 포함되어야 합니다."
        )

    # 같은 날짜의 대표가격이 중복되면 모델 입력을 신뢰할 수 없다.
    if source["price_date"].duplicated().any():
        raise ValueError(
            "대표가격 이력에 중복된 날짜가 있습니다."
        )

    source = source.sort_values(
        "price_date",
        kind="stable",
    ).reset_index(drop=True)

    dates = source["price_date"]
    values = source[
        "representative_standard_unit_price"
    ].to_numpy(dtype=float)

    base_date = dates.iloc[-1]

    # 학습 feature가 최대 28일 과거값을 사용하므로
    # 기준일에서 최소 28일 이전 관측이 존재해야 한다.
    history_start = base_date - pd.Timedelta(days=28)

    if dates.iloc[0] > history_start or len(source) < 2:
        raise InsufficientPriceHistoryError(
            "주간 예측 feature 생성에 필요한 28일 이력이 부족합니다."
        )

    current_price = float(values[-1])

    if current_price <= 0:
        raise ValueError(
            "현재 대표가격은 0보다 커야 합니다."
        )

    feature_row: dict[str, object] = {
        "series_id": int(source["series_id"].iloc[-1]),
        "ingredient_code": str(
            source["ingredient_code"].iloc[-1]
        ),
        "base_date": base_date,
        "current_price": current_price,
        "lag_1_observation": float(values[-2]),
        "day_of_week": base_date.dayofweek,
        "month": base_date.month,
        "season": _season_for_month(base_date.month),
    }

    # 학습 시와 동일하게 7/14/28일 window feature를 계산한다.
    for days in (7, 14, 28):
        window_start = (
            base_date - pd.Timedelta(days=days - 1)
        )

        window_mask = dates.between(
            window_start,
            base_date,
        )

        window_values = values[
            window_mask.to_numpy()
        ]

        if len(window_values) == 0:
            raise InsufficientPriceHistoryError(
                f"최근 {days}일 가격 이력이 없습니다."
            )

        lag_value = _last_price_on_or_before(
            dates,
            values,
            base_date - pd.Timedelta(days=days),
        )

        if not np.isfinite(lag_value) or lag_value <= 0:
            raise InsufficientPriceHistoryError(
                f"lag_{days}d 계산에 필요한 가격 이력이 부족합니다."
            )

        feature_row[f"lag_{days}d"] = lag_value
        feature_row[f"mean_{days}d"] = float(
            np.mean(window_values)
        )
        feature_row[f"std_{days}d"] = float(
            np.std(window_values, ddof=0)
        )
        feature_row[f"return_{days}d"] = (
            current_price / lag_value - 1
        )

    lag_1 = float(feature_row["lag_1_observation"])

    if lag_1 <= 0:
        raise ValueError(
            "직전 대표가격은 0보다 커야 합니다."
        )

    # 직전 관측 대비 변화율
    feature_row["return_1_observation"] = (
        current_price / lag_1 - 1
    )

    # 단기 평균과 중기 평균의 상대 변화
    feature_row["short_trend"] = (
        float(feature_row["mean_7d"])
        / float(feature_row["mean_14d"])
        - 1
    )

    # 단기 평균과 장기 평균의 상대 변화
    feature_row["long_trend"] = (
        float(feature_row["mean_7d"])
        / float(feature_row["mean_28d"])
        - 1
    )

    result = pd.DataFrame([feature_row])

    # 모델에 실제 전달되는 feature가 모두 정상 값인지 검증한다.
    required_features = [
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]

    if result[required_features].isna().any().any():
        raise InsufficientPriceHistoryError(
            "생성된 모델 feature에 결측값이 있습니다."
        )

    numeric_values = result[
        list(NUMERIC_FEATURES)
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "생성된 모델 feature에 무한대 또는 비정상 값이 있습니다."
        )

    return result


def _last_price_on_or_before(
    dates: pd.Series,
    values: np.ndarray,
    cutoff: pd.Timestamp,
) -> float:
    """
    cutoff 날짜 또는 그 이전에서 가장 최근에 관측된 가격을 반환한다.

    KAMIS는 매일 가격이 존재하지 않을 수 있으므로
    정확히 N일 전 데이터가 없어도 가장 가까운 이전 관측을 사용한다.
    """
    positions = np.flatnonzero(
        dates.le(cutoff).to_numpy()
    )

    if len(positions) == 0:
        return np.nan

    return float(values[positions[-1]])


def _season_for_month(month: int) -> str:
    """학습 코드와 동일한 월별 계절 구분."""

    if month in (3, 4, 5):
        return "SPRING"

    if month in (6, 7, 8):
        return "SUMMER"

    if month in (9, 10, 11):
        return "AUTUMN"

    return "WINTER"