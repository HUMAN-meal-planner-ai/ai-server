from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from training.price.compare_models import (
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_SNAPSHOT,
    SEQUENCE_LENGTH,
    _load_snapshot,
)
from training.price.evaluation_metrics import calculate_metrics
from training.price.feature_engineering import TARGET_COLUMN
from training.price.model_definitions import build_preprocessor, build_tabular_models
from training.price.sequence_builder import sequence_eligible_indices
from training.price.time_split import determine_time_boundaries, split_name_for_dates


CANDIDATE_MODELS = ("baseline_lag_1", "ridge", "gradient_boosting")
ROLLING_FOLD_COUNT = 3
LARGE_CHANGE_QUANTILE = 0.75
CHANGE_TOLERANCE = 1e-12


def main() -> None:
    frame = _load_snapshot(DEFAULT_SNAPSHOT)
    boundaries = determine_time_boundaries(frame)
    frame["split"] = split_name_for_dates(frame["price_date"], boundaries)
    eligible_indices = sequence_eligible_indices(frame, SEQUENCE_LENGTH)
    validation_indices = np.asarray(
        [index for index in eligible_indices if frame.at[index, "split"] == "validation"],
        dtype=int,
    )

    predictions = _fit_and_predict(frame, eligible_indices, validation_indices)
    annotated, large_change_threshold = _annotate_price_changes(predictions)
    segment_metrics = _calculate_segment_metrics(annotated)
    series_metrics = _calculate_series_metrics(annotated)
    rolling_metrics = _run_rolling_backtest(frame, eligible_indices)

    _write_results(
        frame,
        boundaries,
        segment_metrics,
        series_metrics,
        rolling_metrics,
        large_change_threshold,
    )
    print(segment_metrics.to_string(index=False))
    print("\nValidation expanding-window backtest")
    print(rolling_metrics.to_string(index=False))


def _fit_and_predict(
    frame: pd.DataFrame,
    eligible_indices: np.ndarray,
    evaluation_indices: np.ndarray,
) -> pd.DataFrame:
    train_indices = np.asarray(
        [index for index in eligible_indices if frame.at[index, "split"] == "train"],
        dtype=int,
    )
    preprocessor = build_preprocessor()
    preprocessor.fit(frame[frame["split"] == "train"])
    transformed = np.asarray(preprocessor.transform(frame), dtype=np.float32)
    actual = frame.loc[evaluation_indices, TARGET_COLUMN].to_numpy(dtype=float)
    rows = [
        _prediction_frame(
            "baseline_lag_1",
            frame,
            evaluation_indices,
            actual,
            frame.loc[evaluation_indices, "lag_1"].to_numpy(dtype=float),
        )
    ]

    models = build_tabular_models()
    for model_name in ("ridge", "gradient_boosting"):
        model = models[model_name]
        model.fit(
            transformed[train_indices],
            frame.loc[train_indices, TARGET_COLUMN].to_numpy(dtype=float),
        )
        predicted = model.predict(transformed[evaluation_indices])
        rows.append(
            _prediction_frame(
                model_name, frame, evaluation_indices, actual, predicted
            )
        )
    return pd.concat(rows, ignore_index=True)


def _prediction_frame(model_name, frame, indices, actual, predicted) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model_name,
            "series_id": frame.loc[indices, "series_id"].to_numpy(),
            "price_date": frame.loc[indices, "price_date"].to_numpy(),
            "actual": actual,
            "lag_1": frame.loc[indices, "lag_1"].to_numpy(dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
        }
    )


def _annotate_price_changes(predictions: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    annotated = predictions.copy()
    annotated["absolute_change"] = np.abs(annotated["actual"] - annotated["lag_1"])
    annotated["is_price_change"] = annotated["absolute_change"] > CHANGE_TOLERANCE
    annotated["relative_change_percent"] = np.where(
        annotated["lag_1"] != 0,
        annotated["absolute_change"] / np.abs(annotated["lag_1"]) * 100,
        np.nan,
    )
    baseline_rows = annotated[annotated["model"] == "baseline_lag_1"]
    changed_relative = baseline_rows.loc[
        baseline_rows["is_price_change"] & baseline_rows["relative_change_percent"].notna(),
        "relative_change_percent",
    ]
    if changed_relative.empty:
        raise ValueError("Validation 구간에 가격 변화 행이 없어 변화 성능을 평가할 수 없습니다.")
    threshold = float(changed_relative.quantile(LARGE_CHANGE_QUANTILE))
    annotated["is_large_change"] = (
        annotated["is_price_change"]
        & annotated["relative_change_percent"].ge(threshold)
    )
    return annotated, threshold


def _calculate_segment_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    segments = {
        "all_validation": pd.Series(True, index=predictions.index),
        "price_change": predictions["is_price_change"],
        "large_change_relative_top_25_percent": predictions["is_large_change"],
    }
    rows = []
    for segment_name, mask in segments.items():
        for model_name in CANDIDATE_MODELS:
            group = predictions[mask & predictions["model"].eq(model_name)]
            rows.append(
                {
                    "segment": segment_name,
                    "model": model_name,
                    **_selected_metrics(group["actual"], group["predicted"]),
                }
            )
    return pd.DataFrame(rows)


def _calculate_series_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment_name, mask in (
        ("all_validation", pd.Series(True, index=predictions.index)),
        ("price_change", predictions["is_price_change"]),
        ("large_change_relative_top_25_percent", predictions["is_large_change"]),
    ):
        selected = predictions[mask]
        for (model_name, series_id), group in selected.groupby(
            ["model", "series_id"], sort=True
        ):
            rows.append(
                {
                    "segment": segment_name,
                    "model": model_name,
                    "series_id": series_id,
                    **_selected_metrics(group["actual"], group["predicted"]),
                }
            )
    return pd.DataFrame(rows)


def _run_rolling_backtest(
    frame: pd.DataFrame, eligible_indices: np.ndarray
) -> pd.DataFrame:
    validation_dates = np.asarray(
        sorted(frame.loc[frame["split"] == "validation", "price_date"].unique())
    )
    date_folds = [fold for fold in np.array_split(validation_dates, ROLLING_FOLD_COUNT) if len(fold)]
    prediction_rows = []
    metric_rows = []
    for fold_number, fold_dates in enumerate(date_folds, start=1):
        fold_start = pd.Timestamp(fold_dates[0])
        fold_end = pd.Timestamp(fold_dates[-1])
        train_indices = np.asarray(
            [index for index in eligible_indices if frame.at[index, "price_date"] < fold_start],
            dtype=int,
        )
        evaluation_indices = np.asarray(
            [
                index
                for index in eligible_indices
                if fold_start <= frame.at[index, "price_date"] <= fold_end
                and frame.at[index, "split"] == "validation"
            ],
            dtype=int,
        )
        preprocessor = build_preprocessor()
        preprocessor.fit(frame[frame["price_date"] < fold_start])
        transformed = np.asarray(preprocessor.transform(frame), dtype=np.float32)
        actual = frame.loc[evaluation_indices, TARGET_COLUMN].to_numpy(dtype=float)
        fold_predictions = {
            "baseline_lag_1": frame.loc[evaluation_indices, "lag_1"].to_numpy(dtype=float)
        }
        models = build_tabular_models()
        for model_name in ("ridge", "gradient_boosting"):
            model = models[model_name]
            model.fit(
                transformed[train_indices],
                frame.loc[train_indices, TARGET_COLUMN].to_numpy(dtype=float),
            )
            fold_predictions[model_name] = model.predict(transformed[evaluation_indices])

        for model_name, predicted in fold_predictions.items():
            metrics = _selected_metrics(actual, predicted)
            metric_rows.append(
                {
                    "fold": fold_number,
                    "start_date": fold_start.date().isoformat(),
                    "end_date": fold_end.date().isoformat(),
                    "train_rows": len(train_indices),
                    "model": model_name,
                    **metrics,
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {"model": model_name, "actual": actual, "predicted": predicted}
                )
            )

    pooled = pd.concat(prediction_rows, ignore_index=True)
    for model_name in CANDIDATE_MODELS:
        group = pooled[pooled["model"] == model_name]
        metric_rows.append(
            {
                "fold": "pooled",
                "start_date": pd.Timestamp(validation_dates[0]).date().isoformat(),
                "end_date": pd.Timestamp(validation_dates[-1]).date().isoformat(),
                "train_rows": None,
                "model": model_name,
                **_selected_metrics(group["actual"], group["predicted"]),
            }
        )
    return pd.DataFrame(metric_rows)


def _selected_metrics(actual, predicted) -> dict[str, float | int | None]:
    metrics = calculate_metrics(actual, predicted)
    return {
        "row_count": metrics["row_count"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape_percent": metrics["mape_percent"],
    }


def _write_results(
    frame,
    boundaries,
    segment_metrics,
    series_metrics,
    rolling_metrics,
    large_change_threshold,
) -> None:
    DEFAULT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    maximum_date = frame["price_date"].max().date().isoformat()
    prefix = DEFAULT_OUTPUT_DIRECTORY / f"final_candidate_analysis_as_of_{maximum_date}"
    segment_metrics.to_csv(
        Path(str(prefix) + ".segments.csv"), index=False, encoding="utf-8"
    )
    series_metrics.to_csv(
        Path(str(prefix) + ".series.csv"), index=False, encoding="utf-8"
    )
    rolling_metrics.to_csv(
        Path(str(prefix) + ".rolling.csv"), index=False, encoding="utf-8"
    )
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(DEFAULT_SNAPSHOT),
        "models": list(CANDIDATE_MODELS),
        "validation_start": (
            frame.loc[frame["split"] == "validation", "price_date"].min().date().isoformat()
        ),
        "validation_end": boundaries.validation_end.date().isoformat(),
        "large_change_definition": "가격 변화 행 중 직전 관측값 대비 절대 변화율 상위 25%",
        "large_change_relative_percent_threshold": large_change_threshold,
        "rolling_backtest": "Validation 날짜 3개 연속 구간, expanding-window 재학습",
        "production_model_artifacts_saved": False,
    }
    Path(str(prefix) + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
