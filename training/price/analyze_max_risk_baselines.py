from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score

from training.price.analyze_weekly_forecast import MAX_TARGET_COLUMN
from training.price.compare_regional_weekly_models import (
    DEFAULT_EVALUATION_DIRECTORY,
    DEFAULT_SNAPSHOT,
    build_expanding_folds,
    build_preprocessor,
    load_snapshot,
    select_fold_rows,
)
from training.price.compare_weekly_target_designs import (
    build_prediction_frame as build_model_prediction_frame,
    training_large_rise_threshold,
)


BASELINES = (
    "always_rise",
    "current_price_persistence",
    "recent_7d_max_persistence",
    "return_7d_threshold",
    "return_14d_threshold",
    "volatility_7d_threshold",
    "max_price_ridge",
)


def main() -> None:
    arguments = _parse_arguments()
    frame = load_snapshot(arguments.snapshot)
    if "max_7d" not in frame.columns:
        raise ValueError("최근 최고가 persistence 비교에 필요한 max_7d 컬럼이 없습니다.")
    ridge = build_max_ridge_predictions(frame)

    prediction_frames = []
    threshold_rows = []
    for fold_number, fold_ridge in ridge.groupby("fold", sort=True):
        evaluation_start = fold_ridge["base_date"].min()
        evaluation_end = fold_ridge["base_date"].max()
        train, evaluation = select_fold_rows(frame, evaluation_start, evaluation_end)
        evaluation = align_evaluation(evaluation, fold_ridge)
        large_rise_threshold = float(fold_ridge["large_rise_threshold"].iloc[0])
        train_return = (
            train[MAX_TARGET_COLUMN].to_numpy(dtype=float)
            / train["current_price"].to_numpy(dtype=float)
            - 1
        )
        train_actual = train_return >= large_rise_threshold

        signals = {
            "return_7d_threshold": train["return_7d"].to_numpy(dtype=float),
            "return_14d_threshold": train["return_14d"].to_numpy(dtype=float),
            "volatility_7d_threshold": (
                train["std_7d"].to_numpy(dtype=float)
                / train["current_price"].to_numpy(dtype=float)
            ),
        }
        selected_thresholds = {}
        for baseline, signal in signals.items():
            threshold, metrics = select_signal_threshold(signal, train_actual)
            selected_thresholds[baseline] = threshold
            threshold_rows.append(
                {
                    "fold": int(fold_number),
                    "baseline": baseline,
                    "selected_threshold": threshold,
                    **metrics,
                }
            )

        current = evaluation["current_price"].to_numpy(dtype=float)
        evaluation_signals = {
            "return_7d_threshold": evaluation["return_7d"].to_numpy(dtype=float),
            "return_14d_threshold": evaluation["return_14d"].to_numpy(dtype=float),
            "volatility_7d_threshold": (
                evaluation["std_7d"].to_numpy(dtype=float) / current
            ),
        }
        predicted = {
            "always_rise": np.ones(len(evaluation), dtype=bool),
            "current_price_persistence": np.zeros(len(evaluation), dtype=bool),
            "recent_7d_max_persistence": (
                evaluation["max_7d"].to_numpy(dtype=float) / current - 1
            )
            >= large_rise_threshold,
            "max_price_ridge": fold_ridge["predicted_large_rise"].astype(bool).to_numpy(),
        }
        predicted_rise = {
            "always_rise": np.ones(len(evaluation), dtype=bool),
            "current_price_persistence": np.zeros(len(evaluation), dtype=bool),
            "recent_7d_max_persistence": evaluation["max_7d"].to_numpy(dtype=float)
            > current,
            "max_price_ridge": fold_ridge["predicted_return"].to_numpy(dtype=float) > 0,
        }
        for baseline, signal in evaluation_signals.items():
            predicted[baseline] = signal >= selected_thresholds[baseline]
            predicted_rise[baseline] = predicted[baseline]

        for baseline in BASELINES:
            prediction_frames.append(
                build_prediction_frame(
                    int(fold_number),
                    baseline,
                    evaluation,
                    fold_ridge,
                    predicted[baseline],
                    predicted_rise[baseline],
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    overall = calculate_detection_metrics(predictions, group_columns=["baseline"])
    by_fold = calculate_detection_metrics(
        predictions, group_columns=["fold", "baseline"]
    )
    write_results(
        arguments.output_directory,
        predictions,
        overall,
        by_fold,
        pd.DataFrame(threshold_rows),
        arguments.snapshot,
    )
    print(overall.to_string(index=False))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="7일 최대가격 Ridge와 단순 위험 baseline 비교")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_EVALUATION_DIRECTORY
    )
    return parser.parse_args()


def build_max_ridge_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    prediction_frames = []
    for fold_number, evaluation_start, evaluation_end in build_expanding_folds(frame):
        train, evaluation = select_fold_rows(frame, evaluation_start, evaluation_end)
        preprocessor = build_preprocessor()
        preprocessor.fit(train)
        x_train = np.asarray(preprocessor.transform(train), dtype=np.float32)
        x_evaluation = np.asarray(
            preprocessor.transform(evaluation), dtype=np.float32
        )
        model = Ridge(alpha=1.0)
        model.fit(x_train, train[MAX_TARGET_COLUMN].to_numpy(dtype=float))
        prediction_frames.append(
            build_model_prediction_frame(
                MAX_TARGET_COLUMN,
                "ridge",
                fold_number,
                evaluation,
                evaluation[MAX_TARGET_COLUMN].to_numpy(dtype=float),
                model.predict(x_evaluation),
                training_large_rise_threshold(train, MAX_TARGET_COLUMN),
            )
        )
    return pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["fold", "base_date", "series_id"], kind="stable"
    )


def align_evaluation(evaluation: pd.DataFrame, ridge: pd.DataFrame) -> pd.DataFrame:
    keys = ["series_id", "base_date"]
    aligned = ridge[keys].merge(evaluation, on=keys, how="left", validate="one_to_one")
    if aligned[MAX_TARGET_COLUMN].isna().any():
        raise ValueError("Ridge 평가 표본과 현재 snapshot의 기준일이 일치하지 않습니다.")
    return aligned


def select_signal_threshold(
    signal: np.ndarray, actual: np.ndarray
) -> tuple[float, dict[str, float]]:
    finite = signal[np.isfinite(signal)]
    if not len(finite):
        raise ValueError("위험 baseline threshold를 선택할 유효 신호가 없습니다.")
    candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 201)))
    scored = []
    for threshold in candidates:
        metrics = binary_metrics(actual, signal >= threshold)
        scored.append((metrics["f1_percent"], metrics["balanced_accuracy_percent"], threshold, metrics))
    _, _, threshold, metrics = max(scored, key=lambda row: (row[0], row[1], row[2]))
    return float(threshold), metrics


def build_prediction_frame(
    fold: int,
    baseline: str,
    evaluation: pd.DataFrame,
    ridge: pd.DataFrame,
    predicted_large_rise: np.ndarray,
    predicted_rise: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": fold,
            "baseline": baseline,
            "series_id": evaluation["series_id"].to_numpy(),
            "base_date": evaluation["base_date"].to_numpy(),
            "actual_large_rise": ridge["is_large_rise"].astype(bool).to_numpy(),
            "predicted_large_rise": np.asarray(predicted_large_rise, dtype=bool),
            "actual_rise": ridge["actual_return"].to_numpy(dtype=float) > 0,
            "predicted_rise": np.asarray(predicted_rise, dtype=bool),
        }
    )


def binary_metrics(actual, predicted) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    true_positive = int((actual & predicted).sum())
    false_positive = int((~actual & predicted).sum())
    true_negative = int((~actual & ~predicted).sum())
    false_negative = int((actual & ~predicted).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "row_count": int(len(actual)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision_percent": precision * 100,
        "recall_percent": recall * 100,
        "f1_percent": f1 * 100,
        "balanced_accuracy_percent": float(
            balanced_accuracy_score(actual, predicted) * 100
        ),
        "warning_rate_percent": float(predicted.mean() * 100),
    }


def calculate_detection_metrics(
    predictions: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(group_columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, keys))
        risk = binary_metrics(group["actual_large_rise"], group["predicted_large_rise"])
        actual_rise = group["actual_rise"]
        predicted_rise = group["predicted_rise"]
        rows.append(
            {
                **identity,
                **risk,
                "actual_large_rise_rate_percent": float(
                    group["actual_large_rise"].mean() * 100
                ),
                "predicted_any_rise_rate_percent": float(predicted_rise.mean() * 100),
                "direction_rise_true_positive": int((actual_rise & predicted_rise).sum()),
                "direction_rise_false_positive": int((~actual_rise & predicted_rise).sum()),
                "direction_fall_true_negative": int((~actual_rise & ~predicted_rise).sum()),
                "direction_rise_false_negative": int((actual_rise & ~predicted_rise).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_results(
    output_directory: Path,
    predictions: pd.DataFrame,
    overall: pd.DataFrame,
    by_fold: pd.DataFrame,
    thresholds: pd.DataFrame,
    snapshot_path: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    maximum_date = predictions["base_date"].max().date().isoformat()
    prefix = output_directory / f"max_risk_baseline_comparison_as_of_{maximum_date}"
    predictions.to_csv(Path(str(prefix) + ".predictions.csv"), index=False, date_format="%Y-%m-%d", encoding="utf-8")
    overall.to_csv(Path(str(prefix) + ".overall.csv"), index=False, encoding="utf-8")
    by_fold.to_csv(Path(str(prefix) + ".folds.csv"), index=False, encoding="utf-8")
    thresholds.to_csv(Path(str(prefix) + ".thresholds.csv"), index=False, encoding="utf-8")
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(snapshot_path),
        "ridge_predictions": "source snapshot에서 outer fold별로 재생성",
        "positive_label": "기존 실험과 동일한 fold별 next_7d_max_price 큰 상승",
        "threshold_selection": "각 outer fold train에서 F1, balanced accuracy 순으로 선택",
        "evaluation_sample_changed": False,
        "model_artifacts_saved": False,
        "production_api_changed": False,
        "production_database_written": False,
    }
    Path(str(prefix) + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
