from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from training.price.analyze_weekly_forecast import (
    CHANGE_TOLERANCE,
    LARGE_CHANGE_QUANTILE,
    TARGET_COLUMN,
)
from training.price.build_regional_weekly_dataset import DEFAULT_OUTPUT_DIRECTORY
from training.price.evaluation_metrics import calculate_metrics


DEFAULT_SNAPSHOT = (
    DEFAULT_OUTPUT_DIRECTORY / "regional_weekly_features_as_of_2026-09-15.csv"
)
DEFAULT_EVALUATION_DIRECTORY = Path(__file__).resolve().parent / "evaluation"
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
CATEGORICAL_FEATURES = ("season", "series_id", "ingredient_code")
MODEL_NAMES = ("baseline_lag_1", "ridge", "xgboost", "gradient_boosting")
OUTER_FOLD_COUNT = 4
INITIAL_TRAIN_FRACTION = 0.70
RANDOM_SEED = 42
XGBOOST_CONFIGS = (
    {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.03,
        "n_estimators": 250,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
    },
)


def main() -> None:
    arguments = _parse_arguments()
    frame = load_snapshot(arguments.snapshot)
    folds = build_expanding_folds(frame)

    prediction_frames = []
    tuning_rows = []
    timing_rows = []
    for fold_number, evaluation_start, evaluation_end in folds:
        train, evaluation = select_fold_rows(frame, evaluation_start, evaluation_end)
        threshold = training_large_change_threshold(train)
        preprocessor = build_preprocessor()
        fit_start = time.perf_counter()
        preprocessor.fit(train)
        transformed_train = np.asarray(preprocessor.transform(train), dtype=np.float32)
        transformed_evaluation = np.asarray(
            preprocessor.transform(evaluation), dtype=np.float32
        )
        preprocessing_seconds = time.perf_counter() - fit_start

        actual = evaluation[TARGET_COLUMN].to_numpy(dtype=float)
        baseline = evaluation["current_price"].to_numpy(dtype=float)
        prediction_frames.append(
            build_prediction_frame(
                "baseline_lag_1", fold_number, evaluation, actual, baseline, threshold
            )
        )
        timing_rows.append(
            _timing_row(
                fold_number,
                "baseline_lag_1",
                len(train),
                len(evaluation),
                0.0,
                0.0,
                preprocessing_seconds,
            )
        )

        models = {
            "ridge": Ridge(alpha=1.0),
            "gradient_boosting": GradientBoostingRegressor(
                learning_rate=0.05,
                n_estimators=200,
                max_depth=3,
                min_samples_leaf=10,
                random_state=RANDOM_SEED,
            ),
        }
        selected_config, fold_tuning_rows = select_xgboost_config(train, fold_number)
        tuning_rows.extend(fold_tuning_rows)
        models["xgboost"] = build_xgboost(selected_config)

        y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
        for model_name, model in models.items():
            fit_start = time.perf_counter()
            model.fit(transformed_train, y_train)
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            predicted = model.predict(transformed_evaluation)
            prediction_seconds = time.perf_counter() - predict_start
            prediction_frames.append(
                build_prediction_frame(
                    model_name,
                    fold_number,
                    evaluation,
                    actual,
                    predicted,
                    threshold,
                )
            )
            timing_rows.append(
                _timing_row(
                    fold_number,
                    model_name,
                    len(train),
                    len(evaluation),
                    fit_seconds,
                    prediction_seconds,
                    preprocessing_seconds,
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    segments = calculate_segment_metrics(predictions)
    directions = calculate_direction_metrics(predictions)
    risks = calculate_risk_metrics(predictions)
    series_metrics = calculate_series_metrics(predictions)
    stability = summarize_series_stability(series_metrics)
    write_results(
        arguments.output_directory,
        frame,
        folds,
        predictions,
        segments,
        directions,
        risks,
        series_metrics,
        stability,
        pd.DataFrame(tuning_rows),
        pd.DataFrame(timing_rows),
        arguments.snapshot,
    )
    print("전체 rolling backtest 성능")
    print(segments[segments["segment"] == "all"].to_string(index=False))
    print("\n큰 변동 탐지 성능")
    print(risks.to_string(index=False))
    print("\nseries별 안정성")
    print(stability.to_string(index=False))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3지역 대표가격 주간 운영 모델 비교")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_EVALUATION_DIRECTORY
    )
    return parser.parse_args()


def load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("3지역 대표가격 학습 snapshot을 찾을 수 없습니다: " + str(path))
    frame = pd.read_csv(
        path,
        parse_dates=["base_date", "target_end_date"],
    )
    required = {
        "series_id",
        "ingredient_code",
        "base_date",
        "target_end_date",
        TARGET_COLUMN,
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("운영 모델 비교에 필요한 컬럼이 없습니다: " + ", ".join(missing))
    if frame.duplicated(["series_id", "base_date"]).any():
        raise ValueError("운영 모델 비교 snapshot에 중복된 시계열 기준일이 있습니다.")
    return frame.sort_values(["base_date", "series_id"], kind="stable").reset_index(
        drop=True
    )


def build_expanding_folds(
    frame: pd.DataFrame,
    fold_count: int = OUTER_FOLD_COUNT,
    initial_train_fraction: float = INITIAL_TRAIN_FRACTION,
) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    dates = pd.DatetimeIndex(frame["base_date"].drop_duplicates().sort_values())
    first_evaluation_index = int(len(dates) * initial_train_fraction)
    evaluation_dates = dates[first_evaluation_index:]
    chunks = [chunk for chunk in np.array_split(evaluation_dates, fold_count) if len(chunk)]
    return [
        (number, pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1]))
        for number, chunk in enumerate(chunks, start=1)
    ]


def select_fold_rows(
    frame: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 학습 label의 마지막 날짜가 평가 시작일 전에 끝나도록 7일 target 경계를 지킨다.
    train = frame[frame["target_end_date"] < evaluation_start]
    evaluation = frame[frame["base_date"].between(evaluation_start, evaluation_end)]
    if train.empty or evaluation.empty:
        raise ValueError("rolling backtest의 학습 또는 평가 데이터가 비어 있습니다.")
    return train, evaluation


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


def select_xgboost_config(
    outer_train: pd.DataFrame, fold_number: int
) -> tuple[dict[str, float | int], list[dict[str, float | int | bool]]]:
    dates = pd.DatetimeIndex(outer_train["base_date"].drop_duplicates().sort_values())
    validation_start = dates[int(len(dates) * 0.85)]
    inner_train = outer_train[outer_train["target_end_date"] < validation_start]
    inner_validation = outer_train[outer_train["base_date"] >= validation_start]
    if inner_train.empty or inner_validation.empty:
        raise ValueError("XGBoost 설정 선택용 내부 학습 또는 검증 데이터가 비어 있습니다.")

    preprocessor = build_preprocessor()
    preprocessor.fit(inner_train)
    x_train = np.asarray(preprocessor.transform(inner_train), dtype=np.float32)
    x_validation = np.asarray(preprocessor.transform(inner_validation), dtype=np.float32)
    y_train = inner_train[TARGET_COLUMN].to_numpy(dtype=float)
    y_validation = inner_validation[TARGET_COLUMN].to_numpy(dtype=float)

    rows = []
    for config_number, config in enumerate(XGBOOST_CONFIGS, start=1):
        model = build_xgboost(config)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        predicted = model.predict(x_validation)
        elapsed = time.perf_counter() - started
        metrics = calculate_metrics(y_validation, predicted)
        rows.append(
            {
                "fold": fold_number,
                "config_number": config_number,
                **config,
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "fit_and_predict_seconds": elapsed,
            }
        )
    selected = min(rows, key=lambda row: (row["validation_mae"], row["validation_rmse"]))
    for row in rows:
        row["selected"] = row["config_number"] == selected["config_number"]
    config_keys = XGBOOST_CONFIGS[0].keys()
    return {key: selected[key] for key in config_keys}, rows


def build_xgboost(config: dict[str, float | int]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=4,
        verbosity=0,
        **config,
    )


def training_large_change_threshold(train: pd.DataFrame) -> float:
    changes = (train[TARGET_COLUMN] / train["current_price"] - 1).abs()
    changed = changes[changes > CHANGE_TOLERANCE]
    if changed.empty:
        raise ValueError("학습 구간에 주간 평균가격 변화가 없습니다.")
    return float(changed.quantile(LARGE_CHANGE_QUANTILE))


def build_prediction_frame(
    model_name: str,
    fold_number: int,
    evaluation: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    large_change_threshold: float,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "fold": fold_number,
            "model": model_name,
            "series_id": evaluation["series_id"].to_numpy(),
            "ingredient_code": evaluation["ingredient_code"].to_numpy(),
            "base_date": evaluation["base_date"].to_numpy(),
            "target_end_date": evaluation["target_end_date"].to_numpy(),
            "current_price": evaluation["current_price"].to_numpy(dtype=float),
            "actual": actual,
            "predicted": np.asarray(predicted, dtype=float),
            "large_change_threshold": large_change_threshold,
        }
    )
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


def calculate_segment_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    segments = {
        "all": pd.Series(True, index=predictions.index),
        "price_change": predictions["is_change"],
        "large_change": predictions["is_large_change"],
    }
    rows = []
    for segment, mask in segments.items():
        for model_name, group in predictions[mask].groupby("model", sort=True):
            metrics = calculate_metrics(group["actual"], group["predicted"])
            rows.append(
                {
                    "model": model_name,
                    "segment": segment,
                    **metrics,
                    "direction_accuracy_percent": float(
                        group["direction_correct"].mean() * 100
                    ),
                }
            )
    return pd.DataFrame(rows)


def calculate_risk_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model", sort=True):
        risks = {
            "all_large_changes": (
                group["is_large_change"],
                group["predicted_large_change"],
            ),
            "large_rise": (
                group["actual_change"] >= group["large_change_threshold"],
                group["predicted_change"] >= group["large_change_threshold"],
            ),
            "large_fall": (
                group["actual_change"] <= -group["large_change_threshold"],
                group["predicted_change"] <= -group["large_change_threshold"],
            ),
        }
        for risk_type, (actual, predicted) in risks.items():
            true_positive = int((actual & predicted).sum())
            false_positive = int((~actual & predicted).sum())
            false_negative = int((actual & ~predicted).sum())
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "model": model_name,
                    "risk_type": risk_type,
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "precision_percent": precision * 100,
                    "recall_percent": recall * 100,
                    "f1_percent": f1 * 100,
                }
            )
    return pd.DataFrame(rows)


def calculate_direction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model", sort=True):
        directions = {
            "rise": group["actual_change"] > CHANGE_TOLERANCE,
            "fall": group["actual_change"] < -CHANGE_TOLERANCE,
            "all_changes": group["is_change"],
        }
        for direction, mask in directions.items():
            selected = group[mask]
            rows.append(
                {
                    "model": model_name,
                    "direction": direction,
                    "row_count": int(len(selected)),
                    "accuracy_percent": float(
                        selected["direction_correct"].mean() * 100
                    ),
                }
            )
    return pd.DataFrame(rows)


def calculate_series_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, series_id), group in predictions.groupby(
        ["model", "series_id"], sort=True
    ):
        metrics = calculate_metrics(group["actual"], group["predicted"])
        changed = group[group["is_change"]]
        rows.append(
            {
                "model": model_name,
                "series_id": series_id,
                **metrics,
                "direction_accuracy_percent": float(
                    changed["direction_correct"].mean() * 100
                )
                if not changed.empty
                else None,
            }
        )
    return pd.DataFrame(rows)


def summarize_series_stability(series_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in series_metrics.groupby("model", sort=True):
        rows.append(
            {
                "model": model_name,
                "series_count": int(group["series_id"].nunique()),
                "mae_median": float(group["mae"].median()),
                "mae_p90": float(group["mae"].quantile(0.90)),
                "mae_max": float(group["mae"].max()),
                "mae_std": float(group["mae"].std(ddof=0)),
                "direction_accuracy_median_percent": float(
                    group["direction_accuracy_percent"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _timing_row(
    fold: int,
    model: str,
    train_rows: int,
    evaluation_rows: int,
    fit_seconds: float,
    prediction_seconds: float,
    preprocessing_seconds: float,
) -> dict[str, int | float | str]:
    return {
        "fold": fold,
        "model": model,
        "train_rows": train_rows,
        "evaluation_rows": evaluation_rows,
        "preprocessing_seconds": preprocessing_seconds,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
    }


def write_results(
    output_directory: Path,
    frame: pd.DataFrame,
    folds: list[tuple[int, pd.Timestamp, pd.Timestamp]],
    predictions: pd.DataFrame,
    segments: pd.DataFrame,
    directions: pd.DataFrame,
    risks: pd.DataFrame,
    series_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    tuning: pd.DataFrame,
    timings: pd.DataFrame,
    snapshot_path: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    maximum_date = frame["base_date"].max().date().isoformat()
    prefix = output_directory / f"regional_weekly_model_comparison_as_of_{maximum_date}"
    outputs = {
        ".predictions.csv": predictions,
        ".segments.csv": segments,
        ".directions.csv": directions,
        ".risk.csv": risks,
        ".series.csv": series_metrics,
        ".stability.csv": stability,
        ".xgboost_tuning.csv": tuning,
        ".timings.csv": timings,
    }
    for suffix, data in outputs.items():
        data.to_csv(Path(str(prefix) + suffix), index=False, date_format="%Y-%m-%d", encoding="utf-8")

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(snapshot_path),
        "target_column": TARGET_COLUMN,
        "models": list(MODEL_NAMES),
        "initial_train_fraction": INITIAL_TRAIN_FRACTION,
        "outer_fold_count": len(folds),
        "folds": [
            {
                "fold": number,
                "evaluation_start": start.date().isoformat(),
                "evaluation_end": end.date().isoformat(),
            }
            for number, start, end in folds
        ],
        "preprocessing": "각 outer fold 학습 구간에서만 scaler/one-hot encoder fit",
        "xgboost_selection": "각 outer fold 내부의 마지막 15% 날짜 validation에서 MAE, RMSE 순으로 선택",
        "large_change_threshold": "각 outer fold 학습 구간의 비영(非零) 절대 변화율 75백분위",
        "model_artifacts_saved": False,
        "production_api_changed": False,
        "production_database_written": False,
    }
    Path(str(prefix) + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
