from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training.price.compare_models import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_SNAPSHOT, _load_snapshot
from training.price.evaluation_metrics import calculate_metrics
from training.price.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, _season_for_month
from training.price.model_definitions import build_tabular_models
from training.price.time_split import determine_time_boundaries


HORIZONS = tuple(range(1, 8))
CANDIDATE_MODELS = ("baseline_lag_1", "ridge", "gradient_boosting")
ROLLING_FOLD_COUNT = 3
CURRENT_PRICE_COLUMN = "current_price"
MULTI_HORIZON_TARGET = "future_standard_unit_price"
NUMERIC_FEATURES = tuple(column for column in FEATURE_COLUMNS if column != "season") + (
    CURRENT_PRICE_COLUMN,
)
CATEGORICAL_FEATURES = ("season", "series_id")


def main() -> None:
    frame = _load_snapshot(DEFAULT_SNAPSHOT)
    boundaries = determine_time_boundaries(frame)
    horizon_frame = build_multi_horizon_frame(frame)

    validation_predictions = fit_predict_window(
        horizon_frame,
        train_target_end=boundaries.train_end,
        evaluation_start=boundaries.train_end + pd.Timedelta(days=1),
        evaluation_end=boundaries.validation_end,
        fold="full_validation",
    )
    rolling_predictions = run_rolling_backtest(
        horizon_frame,
        boundaries.train_end,
        boundaries.validation_end,
    )
    validation_metrics = calculate_horizon_metrics(validation_predictions)
    rolling_metrics = calculate_horizon_metrics(rolling_predictions, include_fold=True)
    rolling_pooled_metrics = calculate_horizon_metrics(rolling_predictions)
    selection = calculate_selection_summary(rolling_predictions)
    selected_model = str(selection.iloc[0]["model"])

    write_results(
        frame,
        validation_metrics,
        rolling_metrics,
        rolling_pooled_metrics,
        selection,
        selected_model,
        rolling_predictions,
    )
    print(validation_metrics.to_string(index=False))
    print("\nValidation rolling/backtest 통합 결과")
    print(rolling_pooled_metrics.to_string(index=False))
    print("\n7일 전체 모델 요약")
    print(selection.to_string(index=False))
    print(f"\n7일 전체 성능 기준 최종 후보: {selected_model}")


def build_multi_horizon_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """기준일의 현재·과거 정보에 달력 기준 D+1~D+7 실제 가격만 label로 결합한다."""
    required = {"series_id", "price_date", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("7일 예측 검증에 필요한 컬럼이 없습니다: " + ", ".join(missing))

    source = frame.sort_values(["series_id", "price_date"], kind="stable").copy()
    source["price_date"] = pd.to_datetime(source["price_date"])
    future_prices = source[["series_id", "price_date", TARGET_COLUMN]].rename(
        columns={"price_date": "target_date", TARGET_COLUMN: MULTI_HORIZON_TARGET}
    )
    rows = []
    for horizon in HORIZONS:
        candidates = source.copy()
        candidates["base_date"] = candidates["price_date"]
        candidates["target_date"] = candidates["base_date"] + pd.Timedelta(days=horizon)
        candidates["horizon"] = horizon
        candidates[CURRENT_PRICE_COLUMN] = candidates[TARGET_COLUMN]
        candidates = candidates.drop(columns=[TARGET_COLUMN])
        candidates = candidates.merge(
            future_prices,
            on=["series_id", "target_date"],
            how="inner",
            validate="one_to_one",
        )

        # 달력 특성은 기준일이 아니라 실제 예측 대상 날짜로 다시 계산한다.
        candidates["day_of_week"] = candidates["target_date"].dt.dayofweek
        candidates["month"] = candidates["target_date"].dt.month
        candidates["season"] = candidates["month"].map(_season_for_month)
        rows.append(candidates)

    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(
        ["horizon", "series_id", "base_date"], kind="stable"
    ).reset_index(drop=True)


def fit_predict_window(
    frame: pd.DataFrame,
    train_target_end: pd.Timestamp,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    fold: str | int,
) -> pd.DataFrame:
    prediction_rows = []
    for horizon in HORIZONS:
        horizon_frame = frame[frame["horizon"] == horizon]
        train, evaluation = select_window_rows(
            horizon_frame, train_target_end, evaluation_start, evaluation_end
        )
        if train.empty or evaluation.empty:
            raise ValueError(f"D+{horizon} 학습 또는 평가 데이터가 비어 있습니다.")

        actual = evaluation[MULTI_HORIZON_TARGET].to_numpy(dtype=float)
        model_predictions = {
            "baseline_lag_1": evaluation[CURRENT_PRICE_COLUMN].to_numpy(dtype=float)
        }
        preprocessor = build_multi_horizon_preprocessor()
        preprocessor.fit(train)
        x_train = np.asarray(preprocessor.transform(train), dtype=np.float32)
        x_evaluation = np.asarray(preprocessor.transform(evaluation), dtype=np.float32)
        models = build_tabular_models()
        for model_name in ("ridge", "gradient_boosting"):
            model = models[model_name]
            model.fit(x_train, train[MULTI_HORIZON_TARGET].to_numpy(dtype=float))
            model_predictions[model_name] = model.predict(x_evaluation)

        for model_name, predicted in model_predictions.items():
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "model": model_name,
                        "horizon": horizon,
                        "series_id": evaluation["series_id"].to_numpy(),
                        "base_date": evaluation["base_date"].to_numpy(),
                        "target_date": evaluation["target_date"].to_numpy(),
                        "actual": actual,
                        "predicted": np.asarray(predicted, dtype=float),
                    }
                )
            )
    return pd.concat(prediction_rows, ignore_index=True)


def select_window_rows(
    frame: pd.DataFrame,
    train_target_end: pd.Timestamp,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 학습 label까지 평가 시작 전에 확정된 행만 허용해 horizon 경계 누수를 막는다.
    train = frame[frame["target_date"] <= train_target_end]
    evaluation = frame[
        frame["base_date"].between(evaluation_start, evaluation_end)
        & frame["target_date"].le(evaluation_end)
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


def build_multi_horizon_preprocessor() -> ColumnTransformer:
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


def calculate_horizon_metrics(
    predictions: pd.DataFrame, include_fold: bool = False
) -> pd.DataFrame:
    group_columns = (["fold"] if include_fold else []) + ["model", "horizon"]
    rows = []
    for keys, group in predictions.groupby(group_columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, keys))
        metrics = calculate_metrics(group["actual"], group["predicted"])
        rows.append(
            {
                **identity,
                "row_count": metrics["row_count"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape_percent": metrics["mape_percent"],
            }
        )
    return pd.DataFrame(rows)


def calculate_selection_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    pooled_horizon_metrics = calculate_horizon_metrics(predictions)
    rows = []
    for model_name, group in pooled_horizon_metrics.groupby("model", sort=True):
        pooled = predictions[predictions["model"] == model_name]
        pooled_metrics = calculate_metrics(pooled["actual"], pooled["predicted"])
        rows.append(
            {
                "model": model_name,
                "mean_horizon_mae": float(group["mae"].mean()),
                "mean_horizon_rmse": float(group["rmse"].mean()),
                "mean_horizon_mape_percent": float(group["mape_percent"].mean()),
                "pooled_mae": pooled_metrics["mae"],
                "pooled_rmse": pooled_metrics["rmse"],
                "pooled_mape_percent": pooled_metrics["mape_percent"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_horizon_mae", "mean_horizon_rmse"], kind="stable"
    ).reset_index(drop=True)


def write_results(
    source_frame: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    rolling_pooled_metrics: pd.DataFrame,
    selection: pd.DataFrame,
    selected_model: str,
    rolling_predictions: pd.DataFrame,
) -> None:
    DEFAULT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    maximum_date = source_frame["price_date"].max().date().isoformat()
    prefix = DEFAULT_OUTPUT_DIRECTORY / f"multi_horizon_analysis_as_of_{maximum_date}"
    validation_metrics.to_csv(
        Path(str(prefix) + ".validation.csv"), index=False, encoding="utf-8"
    )
    rolling_metrics.to_csv(
        Path(str(prefix) + ".rolling.csv"), index=False, encoding="utf-8"
    )
    rolling_pooled_metrics.to_csv(
        Path(str(prefix) + ".rolling_pooled.csv"), index=False, encoding="utf-8"
    )
    selection.to_csv(
        Path(str(prefix) + ".selection.csv"), index=False, encoding="utf-8"
    )
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(DEFAULT_SNAPSHOT),
        "horizons": list(HORIZONS),
        "horizon_semantics": "기준일 다음 달력 날짜 D+1~D+7",
        "models": list(CANDIDATE_MODELS),
        "rolling_folds": ROLLING_FOLD_COUNT,
        "selection_metric": "rolling/backtest의 horizon별 MAE 평균, 동률이면 RMSE 평균",
        "selected_model": selected_model,
        "evaluated_rows": int(len(rolling_predictions) / len(CANDIDATE_MODELS)),
        "production_api_changed": False,
        "production_db_written": False,
    }
    Path(str(prefix) + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
