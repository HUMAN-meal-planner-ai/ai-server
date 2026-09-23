from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from training.price.analyze_weekly_forecast import (
    CHANGE_TOLERANCE,
    MAX_TARGET_COLUMN,
    RETURN_TARGET_COLUMN,
    TARGET_COLUMN,
)
from training.price.compare_regional_weekly_models import (
    DEFAULT_EVALUATION_DIRECTORY,
    DEFAULT_SNAPSHOT,
    XGBOOST_CONFIGS,
    build_expanding_folds,
    build_preprocessor,
    build_xgboost,
    load_snapshot,
    select_fold_rows,
)
from training.price.evaluation_metrics import calculate_metrics


TARGETS = (TARGET_COLUMN, MAX_TARGET_COLUMN, RETURN_TARGET_COLUMN)
REGRESSION_MODELS = ("ridge", "xgboost", "gradient_boosting")
TWO_STAGE_MODEL = "two_stage_logistic_ridge"
RANDOM_SEED = 42
LARGE_RISE_QUANTILE = 0.75


def main() -> None:
    arguments = _parse_arguments()
    frame = load_snapshot(arguments.snapshot)
    missing_targets = sorted(set(TARGETS).difference(frame.columns))
    if missing_targets:
        raise ValueError("target 비교에 필요한 컬럼이 없습니다: " + ", ".join(missing_targets))
    folds = build_expanding_folds(frame)

    prediction_frames = []
    tuning_rows = []
    timing_rows = []
    for fold_number, evaluation_start, evaluation_end in folds:
        train, evaluation = select_fold_rows(frame, evaluation_start, evaluation_end)
        preprocessor = build_preprocessor()
        preprocessor.fit(train)
        x_train = np.asarray(preprocessor.transform(train), dtype=np.float32)
        x_evaluation = np.asarray(preprocessor.transform(evaluation), dtype=np.float32)

        for target_column in TARGETS:
            large_rise_threshold = training_large_rise_threshold(train, target_column)
            actual = evaluation[target_column].to_numpy(dtype=float)
            baseline = (
                np.zeros(len(evaluation), dtype=float)
                if target_column == RETURN_TARGET_COLUMN
                else evaluation["current_price"].to_numpy(dtype=float)
            )
            prediction_frames.append(
                build_prediction_frame(
                    target_column,
                    "baseline_lag_1",
                    fold_number,
                    evaluation,
                    actual,
                    baseline,
                    large_rise_threshold,
                )
            )

            selected_config, target_tuning_rows = select_xgboost_config(
                train, target_column, fold_number
            )
            tuning_rows.extend(target_tuning_rows)
            models = {
                "ridge": Ridge(alpha=1.0),
                "xgboost": build_xgboost(selected_config),
                "gradient_boosting": GradientBoostingRegressor(
                    learning_rate=0.05,
                    n_estimators=200,
                    max_depth=3,
                    min_samples_leaf=10,
                    random_state=RANDOM_SEED,
                ),
            }
            y_train = train[target_column].to_numpy(dtype=float)
            for model_name, model in models.items():
                started = time.perf_counter()
                model.fit(x_train, y_train)
                predicted = model.predict(x_evaluation)
                elapsed = time.perf_counter() - started
                prediction_frames.append(
                    build_prediction_frame(
                        target_column,
                        model_name,
                        fold_number,
                        evaluation,
                        actual,
                        predicted,
                        large_rise_threshold,
                    )
                )
                timing_rows.append(
                    {
                        "fold": fold_number,
                        "target": target_column,
                        "model": model_name,
                        "train_rows": len(train),
                        "evaluation_rows": len(evaluation),
                        "fit_and_predict_seconds": elapsed,
                    }
                )

            if target_column == RETURN_TARGET_COLUMN:
                started = time.perf_counter()
                predicted = fit_predict_two_stage(
                    x_train,
                    train[target_column].to_numpy(dtype=float),
                    x_evaluation,
                )
                elapsed = time.perf_counter() - started
                prediction_frames.append(
                    build_prediction_frame(
                        target_column,
                        TWO_STAGE_MODEL,
                        fold_number,
                        evaluation,
                        actual,
                        predicted,
                        large_rise_threshold,
                    )
                )
                timing_rows.append(
                    {
                        "fold": fold_number,
                        "target": target_column,
                        "model": TWO_STAGE_MODEL,
                        "train_rows": len(train),
                        "evaluation_rows": len(evaluation),
                        "fit_and_predict_seconds": elapsed,
                    }
                )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = calculate_target_metrics(predictions)
    directions = calculate_direction_metrics(predictions)
    risks = calculate_large_rise_metrics(predictions)
    series_metrics = calculate_series_metrics(predictions)
    stability = summarize_series_stability(series_metrics)
    write_results(
        arguments.output_directory,
        frame,
        folds,
        predictions,
        metrics,
        directions,
        risks,
        series_metrics,
        stability,
        pd.DataFrame(tuning_rows),
        pd.DataFrame(timing_rows),
        arguments.snapshot,
    )
    print(metrics[metrics["segment"] == "all"].to_string(index=False))
    print("\n큰 상승 탐지")
    print(risks.to_string(index=False))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="주간 가격예측 target 설계 비교")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_EVALUATION_DIRECTORY
    )
    return parser.parse_args()


def target_returns(frame: pd.DataFrame, target_column: str) -> np.ndarray:
    values = frame[target_column].to_numpy(dtype=float)
    if target_column == RETURN_TARGET_COLUMN:
        return values
    return values / frame["current_price"].to_numpy(dtype=float) - 1


def training_large_rise_threshold(train: pd.DataFrame, target_column: str) -> float:
    returns = target_returns(train, target_column)
    rises = returns[returns > CHANGE_TOLERANCE]
    if not len(rises):
        raise ValueError("학습 구간에 주간 가격 상승 표본이 없습니다.")
    return float(np.quantile(rises, LARGE_RISE_QUANTILE))


def select_xgboost_config(
    outer_train: pd.DataFrame,
    target_column: str,
    fold_number: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str | bool]]]:
    dates = pd.DatetimeIndex(outer_train["base_date"].drop_duplicates().sort_values())
    validation_start = dates[int(len(dates) * 0.85)]
    inner_train = outer_train[outer_train["target_end_date"] < validation_start]
    inner_validation = outer_train[outer_train["base_date"] >= validation_start]
    if inner_train.empty or inner_validation.empty:
        raise ValueError("XGBoost target 비교용 내부 학습 또는 검증 데이터가 비어 있습니다.")

    preprocessor = build_preprocessor()
    preprocessor.fit(inner_train)
    x_train = np.asarray(preprocessor.transform(inner_train), dtype=np.float32)
    x_validation = np.asarray(preprocessor.transform(inner_validation), dtype=np.float32)
    y_train = inner_train[target_column].to_numpy(dtype=float)
    y_validation = inner_validation[target_column].to_numpy(dtype=float)
    rows = []
    for config_number, config in enumerate(XGBOOST_CONFIGS, start=1):
        model = build_xgboost(config)
        model.fit(x_train, y_train)
        metrics = calculate_metrics(y_validation, model.predict(x_validation))
        rows.append(
            {
                "fold": fold_number,
                "target": target_column,
                "config_number": config_number,
                **config,
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
            }
        )
    selected = min(rows, key=lambda row: (row["validation_mae"], row["validation_rmse"]))
    for row in rows:
        row["selected"] = row["config_number"] == selected["config_number"]
    return {
        key: selected[key] for key in XGBOOST_CONFIGS[0]
    }, rows


def fit_predict_two_stage(
    x_train: np.ndarray,
    train_returns: np.ndarray,
    x_evaluation: np.ndarray,
) -> np.ndarray:
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=RANDOM_SEED,
    )
    regressor = Ridge(alpha=1.0)
    classifier.fit(x_train, train_returns > CHANGE_TOLERANCE)
    regressor.fit(x_train, train_returns)
    predicts_rise = classifier.predict_proba(x_evaluation)[:, 1] >= 0.5
    predicted_return = regressor.predict(x_evaluation)
    return np.where(
        predicts_rise,
        np.maximum(predicted_return, 0.0),
        np.minimum(predicted_return, 0.0),
    )


def build_prediction_frame(
    target_column: str,
    model_name: str,
    fold_number: int,
    evaluation: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    large_rise_threshold: float,
) -> pd.DataFrame:
    current = evaluation["current_price"].to_numpy(dtype=float)
    actual_return = actual if target_column == RETURN_TARGET_COLUMN else actual / current - 1
    predicted_return = (
        predicted if target_column == RETURN_TARGET_COLUMN else predicted / current - 1
    )
    actual_price = current * (1 + actual_return)
    predicted_price = current * (1 + predicted_return)
    return pd.DataFrame(
        {
            "target": target_column,
            "model": model_name,
            "fold": fold_number,
            "series_id": evaluation["series_id"].to_numpy(),
            "ingredient_code": evaluation["ingredient_code"].to_numpy(),
            "base_date": evaluation["base_date"].to_numpy(),
            "current_price": current,
            "actual": actual,
            "predicted": predicted,
            "actual_price": actual_price,
            "predicted_price": predicted_price,
            "actual_return": actual_return,
            "predicted_return": predicted_return,
            "is_change": np.abs(actual_return) > CHANGE_TOLERANCE,
            "is_large_rise": actual_return >= large_rise_threshold,
            "predicted_large_rise": predicted_return >= large_rise_threshold,
            "direction_correct": np.sign(actual_return) == np.sign(predicted_return),
            "large_rise_threshold": large_rise_threshold,
        }
    )


def calculate_target_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment, mask in {
        "all": pd.Series(True, index=predictions.index),
        "price_change": predictions["is_change"],
        "large_rise": predictions["is_large_rise"],
    }.items():
        selected = predictions[mask]
        for (target, model), group in selected.groupby(["target", "model"], sort=True):
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "segment": segment,
                    **calculate_metrics(group["actual"], group["predicted"]),
                }
            )
    return pd.DataFrame(rows)


def calculate_direction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), group in predictions.groupby(["target", "model"], sort=True):
        for direction, mask in {
            "rise": group["actual_return"] > CHANGE_TOLERANCE,
            "fall": group["actual_return"] < -CHANGE_TOLERANCE,
            "all_changes": group["is_change"],
        }.items():
            selected = group[mask]
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "direction": direction,
                    "row_count": len(selected),
                    "accuracy_percent": float(selected["direction_correct"].mean() * 100),
                }
            )
    return pd.DataFrame(rows)


def calculate_large_rise_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), group in predictions.groupby(["target", "model"], sort=True):
        actual = group["is_large_rise"]
        predicted = group["predicted_large_rise"]
        true_positive = int((actual & predicted).sum())
        false_positive = int((~actual & predicted).sum())
        false_negative = int((actual & ~predicted).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "target": target,
                "model": model,
                "large_rise_rows": int(actual.sum()),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision_percent": precision * 100,
                "recall_percent": recall * 100,
                "f1_percent": f1 * 100,
            }
        )
    return pd.DataFrame(rows)


def calculate_series_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model, series_id), group in predictions.groupby(
        ["target", "model", "series_id"], sort=True
    ):
        metrics = calculate_metrics(group["actual"], group["predicted"])
        changed = group[group["is_change"]]
        rows.append(
            {
                "target": target,
                "model": model,
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
    for (target, model), group in series_metrics.groupby(["target", "model"], sort=True):
        rows.append(
            {
                "target": target,
                "model": model,
                "series_count": group["series_id"].nunique(),
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


def write_results(
    output_directory: Path,
    frame: pd.DataFrame,
    folds,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
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
    prefix = output_directory / f"weekly_target_design_comparison_as_of_{maximum_date}"
    for suffix, data in {
        ".predictions.csv": predictions,
        ".metrics.csv": metrics,
        ".directions.csv": directions,
        ".risk.csv": risks,
        ".series.csv": series_metrics,
        ".stability.csv": stability,
        ".xgboost_tuning.csv": tuning,
        ".timings.csv": timings,
    }.items():
        data.to_csv(Path(str(prefix) + suffix), index=False, date_format="%Y-%m-%d", encoding="utf-8")

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(snapshot_path),
        "targets": list(TARGETS),
        "regression_models": list(REGRESSION_MODELS),
        "two_stage_model": TWO_STAGE_MODEL,
        "outer_folds": [
            {
                "fold": number,
                "evaluation_start": start.date().isoformat(),
                "evaluation_end": end.date().isoformat(),
            }
            for number, start, end in folds
        ],
        "large_rise_threshold": "각 fold 학습 구간의 양(+) 변화율 75백분위",
        "return_mape_warning": "0 또는 0에 가까운 변화율에서는 MAPE가 불안정하므로 MAE·방향·위험 지표를 우선 해석",
        "preprocessing": "각 outer fold train에서만 scaler와 encoder fit",
        "model_artifacts_saved": False,
        "production_api_changed": False,
        "production_database_written": False,
    }
    Path(str(prefix) + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
