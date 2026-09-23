from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from training.price.analyze_max_risk_baselines import (
    align_evaluation,
    binary_metrics,
    build_max_ridge_predictions,
    select_signal_threshold,
)
from training.price.analyze_weekly_forecast import MAX_TARGET_COLUMN
from training.price.compare_regional_weekly_models import (
    DEFAULT_EVALUATION_DIRECTORY,
    DEFAULT_SNAPSHOT,
    build_preprocessor,
    load_snapshot,
    select_fold_rows,
)


STRATEGIES = (
    "ridge_only",
    "volatility_only",
    "ridge_or_volatility",
    "ridge_and_volatility",
    "ridge_volatility_score",
)
TOP_COUNTS = (5, 10)


def main() -> None:
    arguments = _parse_arguments()
    frame = load_snapshot(arguments.snapshot)
    ridge_predictions = build_max_ridge_predictions(frame)

    prediction_frames = []
    threshold_rows = []
    for fold_number, fold_ridge in ridge_predictions.groupby("fold", sort=True):
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
        stored_ridge_return = fold_ridge["predicted_return"].to_numpy(dtype=float)
        if not np.allclose(evaluation_ridge_return, stored_ridge_return, atol=1e-7):
            raise ValueError("재구성한 Ridge 예측값이 기존 평가 결과와 일치하지 않습니다.")

        train_volatility = (
            train["std_7d"].to_numpy(dtype=float)
            / train["current_price"].to_numpy(dtype=float)
        )
        evaluation_volatility = (
            evaluation["std_7d"].to_numpy(dtype=float)
            / evaluation["current_price"].to_numpy(dtype=float)
        )
        train_actual = (
            train[MAX_TARGET_COLUMN].to_numpy(dtype=float)
            / train["current_price"].to_numpy(dtype=float)
            - 1
        ) >= large_rise_threshold

        volatility_threshold, _ = select_signal_threshold(
            train_volatility, train_actual
        )
        train_ridge_rank = empirical_percentile(train_ridge_return, train_ridge_return)
        train_volatility_rank = empirical_percentile(
            train_volatility, train_volatility
        )
        evaluation_ridge_rank = empirical_percentile(
            train_ridge_return, evaluation_ridge_return
        )
        evaluation_volatility_rank = empirical_percentile(
            train_volatility, evaluation_volatility
        )
        train_combined_score = (train_ridge_rank + train_volatility_rank) / 2
        evaluation_combined_score = (
            evaluation_ridge_rank + evaluation_volatility_rank
        ) / 2
        combined_threshold, _ = select_signal_threshold(
            train_combined_score, train_actual
        )

        ridge_alert = stored_ridge_return >= large_rise_threshold
        volatility_alert = evaluation_volatility >= volatility_threshold
        alerts = {
            "ridge_only": ridge_alert,
            "volatility_only": volatility_alert,
            "ridge_or_volatility": ridge_alert | volatility_alert,
            "ridge_and_volatility": ridge_alert & volatility_alert,
            "ridge_volatility_score": evaluation_combined_score >= combined_threshold,
        }
        ranking_scores = {
            "ridge_only": evaluation_ridge_rank,
            "volatility_only": evaluation_volatility_rank,
            "ridge_or_volatility": np.maximum(
                evaluation_ridge_rank, evaluation_volatility_rank
            ),
            "ridge_and_volatility": np.minimum(
                evaluation_ridge_rank, evaluation_volatility_rank
            ),
            "ridge_volatility_score": evaluation_combined_score,
        }
        threshold_rows.append(
            {
                "fold": int(fold_number),
                "large_rise_threshold": large_rise_threshold,
                "volatility_threshold": volatility_threshold,
                "combined_score_threshold": combined_threshold,
            }
        )
        for strategy in STRATEGIES:
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "fold": int(fold_number),
                        "strategy": strategy,
                        "series_id": evaluation["series_id"].to_numpy(),
                        "base_date": evaluation["base_date"].to_numpy(),
                        "actual_large_rise": fold_ridge["is_large_rise"]
                        .astype(bool)
                        .to_numpy(),
                        "predicted_large_rise": alerts[strategy],
                        "ranking_score": ranking_scores[strategy],
                    }
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    overall = calculate_group_metrics(predictions, ["strategy"])
    folds = calculate_group_metrics(predictions, ["fold", "strategy"])
    top_n = calculate_top_n_metrics(predictions, ["strategy"])
    top_n_folds = calculate_top_n_metrics(predictions, ["fold", "strategy"])
    write_results(
        arguments.output_directory,
        predictions,
        overall,
        folds,
        top_n,
        top_n_folds,
        pd.DataFrame(threshold_rows),
        arguments.snapshot,
    )
    print(overall.to_string(index=False))
    print("\n날짜별 Top-N")
    print(top_n.to_string(index=False))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ridge와 최근 변동성 결합 위험 탐지 검증")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_EVALUATION_DIRECTORY
    )
    return parser.parse_args()


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    reference = np.sort(reference[np.isfinite(reference)])
    if not len(reference):
        raise ValueError("위험 점수를 보정할 학습 신호가 없습니다.")
    return np.searchsorted(reference, values, side="right") / len(reference)


def calculate_group_metrics(
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


def calculate_top_n_metrics(
    predictions: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(group_columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, keys))
        date_groups = list(group.groupby("base_date", sort=True))
        dates_with_risk = sum(
            bool(date_group["actual_large_rise"].any())
            for _, date_group in date_groups
        )
        for top_count in TOP_COUNTS:
            precisions = []
            recalls = []
            hit_count = 0
            selected_count = 0
            actual_count = 0
            for _, date_group in date_groups:
                selected = date_group.nlargest(
                    min(top_count, len(date_group)), "ranking_score"
                )
                hits = int(selected["actual_large_rise"].sum())
                actual = int(date_group["actual_large_rise"].sum())
                precisions.append(hits / len(selected))
                if actual:
                    recalls.append(hits / actual)
                hit_count += hits
                selected_count += len(selected)
                actual_count += actual
            rows.append(
                {
                    **identity,
                    "top_n": top_count,
                    "date_count": len(date_groups),
                    "dates_with_large_rise": dates_with_risk,
                    "precision_at_n_percent": float(np.mean(precisions) * 100),
                    "recall_at_n_percent": float(np.mean(recalls) * 100),
                    "micro_precision_at_n_percent": hit_count / selected_count * 100,
                    "micro_recall_at_n_percent": hit_count / actual_count * 100,
                }
            )
    return pd.DataFrame(rows)


def write_results(
    output_directory: Path,
    predictions: pd.DataFrame,
    overall: pd.DataFrame,
    folds: pd.DataFrame,
    top_n: pd.DataFrame,
    top_n_folds: pd.DataFrame,
    thresholds: pd.DataFrame,
    snapshot_path: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    maximum_date = predictions["base_date"].max().date().isoformat()
    prefix = output_directory / f"combined_max_risk_as_of_{maximum_date}"
    outputs = {
        "predictions": predictions,
        "overall": overall,
        "folds": folds,
        "top_n": top_n,
        "top_n_folds": top_n_folds,
        "thresholds": thresholds,
    }
    for suffix, frame in outputs.items():
        frame.to_csv(
            Path(str(prefix) + f".{suffix}.csv"),
            index=False,
            date_format="%Y-%m-%d",
            encoding="utf-8",
        )
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(snapshot_path),
        "ridge_predictions": "source snapshot에서 outer fold별로 재생성",
        "positive_label": "기존 실험과 동일한 fold별 next_7d_max_price 큰 상승",
        "score_calibration": "각 outer fold train 기준 경험적 percentile",
        "combined_score": "Ridge percentile과 변동성 percentile의 동일 가중 평균",
        "top_n_unit": "base_date별 series 위험 순위",
        "recall_at_n": "큰 상승이 한 건 이상인 날짜의 macro 평균",
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
