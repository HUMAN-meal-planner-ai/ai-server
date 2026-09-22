from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training.price.compare_models import DEFAULT_OUTPUT_DIRECTORY
from training.price.config import DatabaseSettings
from training.price.database import open_read_only_connection
from training.price.evaluation_metrics import calculate_metrics
from training.price.feature_engineering import _season_for_month
from training.price.model_definitions import build_tabular_models
from training.price.price_repository import PriceDataRepository
from training.price.time_split import determine_time_boundaries


MINIMUM_OBSERVATIONS = 200
FUTURE_WINDOW_DAYS = 7
MINIMUM_FUTURE_OBSERVATIONS = 3
ROLLING_FOLD_COUNT = 3
LARGE_CHANGE_QUANTILE = 0.75
CHANGE_TOLERANCE = 1e-12
TARGET_COLUMN = "next_7d_mean_price"
CANDIDATE_MODELS = ("baseline_lag_1", "ridge", "gradient_boosting")
NUMERIC_FEATURES = (
    "current_price",
    "lag_1_observation",
    "lag_7d",
    "lag_14d",
    "lag_28d",
    "mean_7d",
    "mean_14d",
    "mean_28d",
    "std_7d",
    "std_14d",
    "std_28d",
    "return_1_observation",
    "return_7d",
    "return_14d",
    "return_28d",
    "short_trend",
    "long_trend",
    "day_of_week",
    "month",
)
CATEGORICAL_FEATURES = ("season", "series_id")


def main() -> None:
    arguments = parse_arguments()
    settings = DatabaseSettings.from_environment(arguments.env_file)
    with open_read_only_connection(settings) as connection:
        repository = PriceDataRepository(connection)
        eligibility = repository.load_series_eligibility()
        prices = repository.load_eligible_prices(MINIMUM_OBSERVATIONS)

    frame = build_weekly_forecast_frame(prices)
    boundaries = determine_time_boundaries(frame.rename(columns={"base_date": "price_date"}))
    validation_predictions = fit_predict_window(
        frame,
        train_target_end=boundaries.train_end,
        evaluation_start=boundaries.train_end + pd.Timedelta(days=1),
        evaluation_end=boundaries.validation_end,
        fold="full_validation",
    )
    rolling_predictions = run_rolling_backtest(
        frame, boundaries.train_end, boundaries.validation_end
    )
    validation_metrics = calculate_segment_metrics(validation_predictions)
    rolling_metrics = calculate_segment_metrics(rolling_predictions, include_fold=True)
    rolling_pooled = calculate_segment_metrics(rolling_predictions)
    risk_metrics = calculate_risk_metrics(rolling_predictions)

    write_results(
        eligibility,
        frame,
        validation_metrics,
        rolling_metrics,
        rolling_pooled,
        risk_metrics,
    )
    print("현재 DB 데이터 충분성")
    print(summarize_eligibility(eligibility).to_string(index=False))
    print("\nValidation rolling/backtest 통합 결과")
    print(rolling_pooled.to_string(index=False))
    print("\n큰 변동 탐지 결과")
    print(risk_metrics.to_string(index=False))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="다음 7일 평균 가격 예측 검증")
    parser.add_argument("--env-file", type=Path)
    return parser.parse_args()


def summarize_eligibility(eligibility: pd.DataFrame) -> pd.DataFrame:
    counts = pd.to_numeric(eligibility["observation_count"])
    return pd.DataFrame(
        [
            {
                "total_series": len(eligibility),
                "series_with_data": int(counts.gt(0).sum()),
                "eligible_series": int(counts.ge(MINIMUM_OBSERVATIONS).sum()),
                "low_density_series": int(counts.between(1, MINIMUM_OBSERVATIONS - 1).sum()),
                "empty_series": int(counts.eq(0).sum()),
                "minimum_first_date": eligibility["first_price_date"].dropna().min(),
                "maximum_last_date": eligibility["last_price_date"].dropna().max(),
            }
        ]
    )


def prepare_eligibility_report(eligibility: pd.DataFrame) -> pd.DataFrame:
    report = eligibility.copy()
    report["observation_count"] = pd.to_numeric(report["observation_count"])
    report["first_price_date"] = pd.to_datetime(report["first_price_date"])
    report["last_price_date"] = pd.to_datetime(report["last_price_date"])
    report["calendar_span_days"] = (
        report["last_price_date"] - report["first_price_date"]
    ).dt.days.add(1)
    report["observation_density_percent"] = (
        report["observation_count"] / report["calendar_span_days"] * 100
    )
    report["eligibility_status"] = np.select(
        [
            report["observation_count"].eq(0),
            report["observation_count"].lt(MINIMUM_OBSERVATIONS),
        ],
        ["EMPTY", "LOW_DENSITY"],
        default="ELIGIBLE",
    )
    return report


def build_weekly_forecast_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """기준일 이전·당일 관측만 feature로, 이후 7일 관측 평균만 target으로 만든다."""
    required = {"series_id", "price_date", "standard_unit_price"}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ValueError("주간 가격 예측에 필요한 컬럼이 없습니다: " + ", ".join(missing))

    source = prices.copy()
    source["price_date"] = pd.to_datetime(source["price_date"])
    source["standard_unit_price"] = pd.to_numeric(
        source["standard_unit_price"], errors="raise"
    )
    source = source.sort_values(["series_id", "price_date"], kind="stable")
    rows = []
    for series_id, group in source.groupby("series_id", sort=True):
        group = group.reset_index(drop=True)
        dates = group["price_date"]
        values = group["standard_unit_price"].to_numpy(dtype=float)
        for index, base_date in enumerate(dates):
            history_start = base_date - pd.Timedelta(days=28)
            if dates.iloc[0] > history_start or index < 1:
                continue
            target_end = base_date + pd.Timedelta(days=FUTURE_WINDOW_DAYS)
            if target_end > dates.iloc[-1]:
                continue
            future_mask = dates.gt(base_date) & dates.le(target_end)
            future_values = values[future_mask.to_numpy()]
            if len(future_values) < MINIMUM_FUTURE_OBSERVATIONS:
                continue

            current_price = values[index]
            feature_row = {
                "series_id": series_id,
                "base_date": base_date,
                "target_end_date": target_end,
                "future_observation_count": len(future_values),
                TARGET_COLUMN: float(np.mean(future_values)),
                "current_price": current_price,
                "lag_1_observation": values[index - 1],
                "day_of_week": base_date.dayofweek,
                "month": base_date.month,
                "season": _season_for_month(base_date.month),
            }
            for days in (7, 14, 28):
                window_mask = dates.between(
                    base_date - pd.Timedelta(days=days - 1), base_date
                )
                window_values = values[window_mask.to_numpy()]
                lag_value = _last_price_on_or_before(
                    dates, values, base_date - pd.Timedelta(days=days)
                )
                feature_row[f"lag_{days}d"] = lag_value
                feature_row[f"mean_{days}d"] = float(np.mean(window_values))
                feature_row[f"std_{days}d"] = float(np.std(window_values, ddof=0))
                feature_row[f"return_{days}d"] = current_price / lag_value - 1

            feature_row["return_1_observation"] = (
                current_price / feature_row["lag_1_observation"] - 1
            )
            feature_row["short_trend"] = (
                feature_row["mean_7d"] / feature_row["mean_14d"] - 1
            )
            feature_row["long_trend"] = (
                feature_row["mean_7d"] / feature_row["mean_28d"] - 1
            )
            rows.append(feature_row)

    result = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=[*NUMERIC_FEATURES, TARGET_COLUMN])
    return result.sort_values(["series_id", "base_date"], kind="stable").reset_index(drop=True)


def _last_price_on_or_before(
    dates: pd.Series, values: np.ndarray, cutoff: pd.Timestamp
) -> float:
    positions = np.flatnonzero(dates.le(cutoff).to_numpy())
    if len(positions) == 0:
        return np.nan
    return float(values[positions[-1]])


def fit_predict_window(
    frame: pd.DataFrame,
    train_target_end: pd.Timestamp,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    fold: str | int,
) -> pd.DataFrame:
    train, evaluation = select_window_rows(
        frame, train_target_end, evaluation_start, evaluation_end
    )
    if train.empty or evaluation.empty:
        raise ValueError("주간 가격 예측 학습 또는 평가 데이터가 비어 있습니다.")

    change_threshold = training_large_change_threshold(train)
    actual = evaluation[TARGET_COLUMN].to_numpy(dtype=float)
    predictions = {
        "baseline_lag_1": evaluation["current_price"].to_numpy(dtype=float)
    }
    preprocessor = build_preprocessor()
    preprocessor.fit(train)
    x_train = np.asarray(preprocessor.transform(train), dtype=np.float32)
    x_evaluation = np.asarray(preprocessor.transform(evaluation), dtype=np.float32)
    models = build_tabular_models()
    for model_name in ("ridge", "gradient_boosting"):
        model = models[model_name]
        model.fit(x_train, train[TARGET_COLUMN].to_numpy(dtype=float))
        predictions[model_name] = model.predict(x_evaluation)

    rows = []
    for model_name, predicted in predictions.items():
        rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "model": model_name,
                    "series_id": evaluation["series_id"].to_numpy(),
                    "base_date": evaluation["base_date"].to_numpy(),
                    "target_end_date": evaluation["target_end_date"].to_numpy(),
                    "current_price": evaluation["current_price"].to_numpy(dtype=float),
                    "actual": actual,
                    "predicted": np.asarray(predicted, dtype=float),
                    "large_change_threshold": change_threshold,
                }
            )
        )
    return annotate_predictions(pd.concat(rows, ignore_index=True))


def select_window_rows(
    frame: pd.DataFrame,
    train_target_end: pd.Timestamp,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # target 평균에 사용된 마지막 날짜까지 학습 경계 안에 있어야 한다.
    train = frame[frame["target_end_date"] <= train_target_end]
    evaluation = frame[
        frame["base_date"].between(evaluation_start, evaluation_end)
        & frame["target_end_date"].le(evaluation_end)
    ]
    return train, evaluation


def run_rolling_backtest(
    frame: pd.DataFrame,
    validation_start_boundary: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    validation_dates = np.asarray(
        sorted(
            frame.loc[
                frame["base_date"].gt(validation_start_boundary)
                & frame["base_date"].le(validation_end),
                "base_date",
            ].unique()
        )
    )
    folds = [fold for fold in np.array_split(validation_dates, ROLLING_FOLD_COUNT) if len(fold)]
    rows = []
    for fold_number, fold_dates in enumerate(folds, start=1):
        fold_start = pd.Timestamp(fold_dates[0])
        fold_end = pd.Timestamp(fold_dates[-1])
        rows.append(
            fit_predict_window(
                frame,
                train_target_end=fold_start - pd.Timedelta(days=1),
                evaluation_start=fold_start,
                evaluation_end=fold_end,
                fold=fold_number,
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def training_large_change_threshold(train: pd.DataFrame) -> float:
    relative_change = (train[TARGET_COLUMN] / train["current_price"] - 1).abs()
    changed = relative_change[relative_change > CHANGE_TOLERANCE]
    if changed.empty:
        raise ValueError("학습 구간에 주간 평균 가격 변동이 없습니다.")
    return float(changed.quantile(LARGE_CHANGE_QUANTILE))


def annotate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["actual_change"] = result["actual"] / result["current_price"] - 1
    result["predicted_change"] = result["predicted"] / result["current_price"] - 1
    result["is_change"] = result["actual_change"].abs() > CHANGE_TOLERANCE
    result["is_large_change"] = (
        result["actual_change"].abs() >= result["large_change_threshold"]
    )
    result["predicted_large_change"] = (
        result["predicted_change"].abs() >= result["large_change_threshold"]
    )
    result["direction_correct"] = (
        np.sign(result["actual_change"]) == np.sign(result["predicted_change"])
    )
    return result


def calculate_segment_metrics(
    predictions: pd.DataFrame, include_fold: bool = False
) -> pd.DataFrame:
    segments = {
        "all": pd.Series(True, index=predictions.index),
        "price_change": predictions["is_change"],
        "large_change": predictions["is_large_change"],
    }
    group_columns = (["fold"] if include_fold else []) + ["model"]
    rows = []
    for segment_name, mask in segments.items():
        selected = predictions[mask]
        for keys, group in selected.groupby(group_columns, sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            identity = dict(zip(group_columns, keys))
            metrics = calculate_metrics(group["actual"], group["predicted"])
            rows.append(
                {
                    **identity,
                    "segment": segment_name,
                    "row_count": metrics["row_count"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape_percent": metrics["mape_percent"],
                    "direction_accuracy_percent": float(
                        group["direction_correct"].mean() * 100
                    ),
                }
            )
    return pd.DataFrame(rows)


def calculate_risk_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model", sort=True):
        actual = group["is_large_change"]
        predicted = group["predicted_large_change"]
        true_positive = int((actual & predicted).sum())
        false_positive = int((~actual & predicted).sum())
        false_negative = int((actual & ~predicted).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "model": model_name,
                "large_change_precision_percent": precision * 100,
                "large_change_recall_percent": recall * 100,
                "large_change_f1_percent": f1 * 100,
            }
        )
    return pd.DataFrame(rows)


def write_results(
    eligibility: pd.DataFrame,
    frame: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    rolling_pooled: pd.DataFrame,
    risk_metrics: pd.DataFrame,
) -> None:
    DEFAULT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    maximum_date = frame["base_date"].max().date().isoformat()
    prefix = DEFAULT_OUTPUT_DIRECTORY / f"weekly_forecast_analysis_as_of_{maximum_date}"
    prepare_eligibility_report(eligibility).to_csv(
        Path(str(prefix) + ".series.csv"), index=False, encoding="utf-8"
    )
    summarize_eligibility(eligibility).to_csv(
        Path(str(prefix) + ".data_sufficiency.csv"), index=False, encoding="utf-8"
    )
    validation_metrics.to_csv(
        Path(str(prefix) + ".validation.csv"), index=False, encoding="utf-8"
    )
    rolling_metrics.to_csv(
        Path(str(prefix) + ".rolling.csv"), index=False, encoding="utf-8"
    )
    rolling_pooled.to_csv(
        Path(str(prefix) + ".rolling_pooled.csv"), index=False, encoding="utf-8"
    )
    risk_metrics.to_csv(
        Path(str(prefix) + ".risk.csv"), index=False, encoding="utf-8"
    )
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "target": "기준일 다음 7일에 공시된 표준 단위 가격의 평균",
        "minimum_future_observations": MINIMUM_FUTURE_OBSERVATIONS,
        "minimum_series_observations": MINIMUM_OBSERVATIONS,
        "feature_columns": [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES],
        "models": list(CANDIDATE_MODELS),
        "rolling_folds": ROLLING_FOLD_COUNT,
        "large_change_threshold": "각 fold 학습 구간 절대 주간 변동률의 75백분위",
        "production_api_changed": False,
        "production_db_written": False,
    }
    Path(str(prefix) + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
