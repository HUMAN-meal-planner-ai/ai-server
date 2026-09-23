from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from training.price.analyze_combined_max_risk import empirical_percentile
from training.price.analyze_max_risk_baselines import (
    align_evaluation,
    binary_metrics,
    build_max_ridge_predictions,
)
from training.price.analyze_weekly_forecast import MAX_TARGET_COLUMN
from training.price.compare_regional_weekly_models import (
    DEFAULT_EVALUATION_DIRECTORY,
    DEFAULT_SNAPSHOT,
    build_preprocessor,
    load_snapshot,
    select_fold_rows,
)


FIXED_THRESHOLDS = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.72,
    0.74,
    0.76,
    0.78,
    0.80,
    0.82,
    0.85,
    0.90,
    0.95,
)
TARGET_RECALLS = (0.60, 0.70, 0.80)


def main() -> None:
    arguments = _parse_arguments()
    frame = load_snapshot(arguments.snapshot)
    ridge_predictions = build_max_ridge_predictions(frame)

    evaluation_frames = []
    operating_frames = []
    operating_thresholds = []
    for fold_number, fold_ridge in ridge_predictions.groupby("fold", sort=True):
        train, evaluation, train_score, evaluation_score, train_actual = build_fold_scores(
            frame, fold_ridge
        )
        actual = fold_ridge["is_large_rise"].astype(bool).to_numpy()
        evaluation_frames.append(
            pd.DataFrame(
                {
                    "fold": int(fold_number),
                    "series_id": evaluation["series_id"].to_numpy(),
                    "base_date": evaluation["base_date"].to_numpy(),
                    "actual_large_rise": actual,
                    "combined_score": evaluation_score,
                }
            )
        )
        for target_recall in TARGET_RECALLS:
            threshold = select_threshold_for_recall(
                train_score, train_actual, target_recall
            )
            train_metrics = binary_metrics(train_actual, train_score >= threshold)
            operating_thresholds.append(
                {
                    "fold": int(fold_number),
                    "target_recall_percent": target_recall * 100,
                    "selected_threshold": threshold,
                    "train_precision_percent": train_metrics["precision_percent"],
                    "train_recall_percent": train_metrics["recall_percent"],
                    "train_warning_rate_percent": train_metrics["warning_rate_percent"],
                }
            )
            operating_frames.append(
                pd.DataFrame(
                    {
                        "fold": int(fold_number),
                        "target_recall_percent": target_recall * 100,
                        "actual_large_rise": actual,
                        "predicted_large_rise": evaluation_score >= threshold,
                    }
                )
            )

    evaluations = pd.concat(evaluation_frames, ignore_index=True)
    fixed_sweep = calculate_fixed_threshold_sweep(evaluations)
    operating_predictions = pd.concat(operating_frames, ignore_index=True)
    operating_overall = calculate_operating_metrics(
        operating_predictions, ["target_recall_percent"]
    )
    operating_folds = calculate_operating_metrics(
        operating_predictions, ["fold", "target_recall_percent"]
    )
    threshold_frame = pd.DataFrame(operating_thresholds)
    threshold_summary = (
        threshold_frame.groupby("target_recall_percent", as_index=False)
        .agg(
            threshold_mean=("selected_threshold", "mean"),
            threshold_min=("selected_threshold", "min"),
            threshold_max=("selected_threshold", "max"),
            train_recall_mean=("train_recall_percent", "mean"),
        )
    )
    operating_overall = operating_overall.merge(
        threshold_summary, on="target_recall_percent", validate="one_to_one"
    )
    write_results(
        arguments.output_directory,
        evaluations,
        fixed_sweep,
        operating_overall,
        operating_folds,
        threshold_frame,
        arguments.snapshot,
    )
    print("고정 threshold sweep")
    print(fixed_sweep.to_string(index=False))
    print("\ntrain 기준 Recall 운영점의 outer-fold 결과")
    print(operating_overall.to_string(index=False))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ridge와 변동성 결합 score threshold 분석")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_EVALUATION_DIRECTORY
    )
    return parser.parse_args()


def build_fold_scores(
    frame: pd.DataFrame, fold_ridge: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    evaluation_start = fold_ridge["base_date"].min()
    evaluation_end = fold_ridge["base_date"].max()
    train, evaluation = select_fold_rows(frame, evaluation_start, evaluation_end)
    evaluation = align_evaluation(evaluation, fold_ridge)
    large_rise_threshold = float(fold_ridge["large_rise_threshold"].iloc[0])

    preprocessor = build_preprocessor()
    preprocessor.fit(train)
    x_train = np.asarray(preprocessor.transform(train), dtype=np.float32)
    x_evaluation = np.asarray(preprocessor.transform(evaluation), dtype=np.float32)
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, train[MAX_TARGET_COLUMN].to_numpy(dtype=float))
    train_ridge_return = (
        ridge.predict(x_train) / train["current_price"].to_numpy(dtype=float) - 1
    )
    evaluation_ridge_return = (
        ridge.predict(x_evaluation)
        / evaluation["current_price"].to_numpy(dtype=float)
        - 1
    )
    if not np.allclose(
        evaluation_ridge_return,
        fold_ridge["predicted_return"].to_numpy(dtype=float),
        atol=1e-7,
    ):
        raise ValueError("재구성한 Ridge 예측값이 기존 평가 결과와 일치하지 않습니다.")

    train_volatility = (
        train["std_7d"].to_numpy(dtype=float)
        / train["current_price"].to_numpy(dtype=float)
    )
    evaluation_volatility = (
        evaluation["std_7d"].to_numpy(dtype=float)
        / evaluation["current_price"].to_numpy(dtype=float)
    )
    train_score = (
        empirical_percentile(train_ridge_return, train_ridge_return)
        + empirical_percentile(train_volatility, train_volatility)
    ) / 2
    evaluation_score = (
        empirical_percentile(train_ridge_return, evaluation_ridge_return)
        + empirical_percentile(train_volatility, evaluation_volatility)
    ) / 2
    train_actual = (
        train[MAX_TARGET_COLUMN].to_numpy(dtype=float)
        / train["current_price"].to_numpy(dtype=float)
        - 1
    ) >= large_rise_threshold
    return train, evaluation, train_score, evaluation_score, train_actual


def select_threshold_for_recall(
    scores: np.ndarray, actual: np.ndarray, target_recall: float
) -> float:
    if not 0 < target_recall <= 1:
        raise ValueError("목표 Recall은 0보다 크고 1 이하여야 합니다.")
    positive_scores = np.sort(np.asarray(scores, dtype=float)[np.asarray(actual, dtype=bool)])
    if not len(positive_scores):
        raise ValueError("threshold를 선택할 큰 상승 학습 표본이 없습니다.")
    required = int(np.ceil(target_recall * len(positive_scores)))
    return float(positive_scores[len(positive_scores) - required])


def calculate_fixed_threshold_sweep(evaluations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in FIXED_THRESHOLDS:
        rows.append(
            {
                "threshold": threshold,
                **binary_metrics(
                    evaluations["actual_large_rise"],
                    evaluations["combined_score"] >= threshold,
                ),
            }
        )
    return pd.DataFrame(rows)


def calculate_operating_metrics(
    predictions: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(group_columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                **binary_metrics(
                    group["actual_large_rise"], group["predicted_large_rise"]
                ),
            }
        )
    return pd.DataFrame(rows)


def write_results(
    output_directory: Path,
    evaluations: pd.DataFrame,
    fixed_sweep: pd.DataFrame,
    operating_overall: pd.DataFrame,
    operating_folds: pd.DataFrame,
    operating_thresholds: pd.DataFrame,
    snapshot_path: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    maximum_date = evaluations["base_date"].max().date().isoformat()
    prefix = output_directory / f"combined_risk_thresholds_as_of_{maximum_date}"
    outputs = {
        "fixed_sweep": fixed_sweep,
        "recall_operating_points": operating_overall,
        "recall_operating_folds": operating_folds,
        "recall_thresholds": operating_thresholds,
    }
    for suffix, output in outputs.items():
        output.to_csv(
            Path(str(prefix) + f".{suffix}.csv"), index=False, encoding="utf-8"
        )
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(snapshot_path),
        "ridge_predictions": "source snapshot에서 outer fold별로 재생성",
        "fixed_sweep_usage": "평가 민감도 분석 전용이며 운영 threshold 선택에 직접 사용하지 않음",
        "recall_threshold_selection": "각 outer fold train에서 목표 Recall을 만족하는 가장 높은 threshold",
        "ranking_strategy_changed": False,
        "model_artifacts_saved": False,
        "production_api_changed": False,
        "production_database_written": False,
    }
    Path(str(prefix) + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
