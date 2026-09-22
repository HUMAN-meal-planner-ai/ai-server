from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from training.price.evaluation_metrics import (
    calculate_metrics,
    calculate_series_metrics,
)
from training.price.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from training.price.model_definitions import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    build_recurrent_model,
    build_tabular_models,
)
from training.price.sequence_builder import build_sequences, sequence_eligible_indices
from training.price.time_split import determine_time_boundaries, split_name_for_dates


DEFAULT_SNAPSHOT = (
    Path(__file__).resolve().parent
    / "snapshots"
    / "price_features_as_of_2026-09-18.csv"
)
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "evaluation"
SEQUENCE_LENGTH = 30
RANDOM_SEED = 42
DEFAULT_EVALUATION_SPLIT = "validation"


def main() -> None:
    arguments = _parse_arguments()
    frame = _load_snapshot(arguments.snapshot)
    boundaries = determine_time_boundaries(frame)
    frame["split"] = split_name_for_dates(frame["price_date"], boundaries)

    # 범주 목록과 스케일 통계는 train 구간에서만 학습한다.
    preprocessor = build_preprocessor()
    train_frame = frame[frame["split"] == "train"]
    preprocessor.fit(train_frame)
    transformed = np.asarray(preprocessor.transform(frame), dtype=np.float32)

    eligible_indices = sequence_eligible_indices(frame, SEQUENCE_LENGTH)
    split_indices = {
        split_name: np.asarray(
            [index for index in eligible_indices if frame.at[index, "split"] == split_name],
            dtype=int,
        )
        for split_name in ("train", "validation", "test")
    }
    _validate_split_coverage(frame, split_indices)

    predictions = []
    summaries = []
    evaluation_split = arguments.evaluation_split
    evaluation_indices = split_indices[evaluation_split]
    actual_evaluation = frame.loc[evaluation_indices, TARGET_COLUMN].to_numpy(dtype=float)

    baseline_start = time.perf_counter()
    baseline_prediction = frame.loc[evaluation_indices, "lag_1"].to_numpy(dtype=float)
    baseline_prediction_seconds = time.perf_counter() - baseline_start
    predictions.append(
        _prediction_frame("baseline_lag_1", frame, evaluation_indices, baseline_prediction)
    )
    summaries.append(
        _summary_row(
            "baseline_lag_1", actual_evaluation, baseline_prediction, 0.0,
            baseline_prediction_seconds, split_indices, evaluation_split, None,
        )
    )

    x_train = transformed[split_indices["train"]]
    y_train = frame.loc[split_indices["train"], TARGET_COLUMN].to_numpy(dtype=float)
    x_evaluation = transformed[evaluation_indices]
    for model_name, model in build_tabular_models().items():
        fit_start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        prediction_start = time.perf_counter()
        predicted = model.predict(x_evaluation)
        prediction_seconds = time.perf_counter() - prediction_start
        predictions.append(_prediction_frame(model_name, frame, evaluation_indices, predicted))
        summaries.append(
            _summary_row(
                model_name, actual_evaluation, predicted, fit_seconds,
                prediction_seconds, split_indices, evaluation_split, None,
            )
        )

    deep_results = _train_recurrent_models(
        frame, transformed, split_indices, evaluation_split
    )
    for model_name, predicted, fit_seconds, prediction_seconds, epochs in deep_results:
        predictions.append(
            _prediction_frame(model_name, frame, evaluation_indices, predicted)
        )
        summaries.append(
            _summary_row(
                model_name, actual_evaluation, predicted, fit_seconds,
                prediction_seconds, split_indices, evaluation_split, epochs,
            )
        )

    prediction_frame = pd.concat(predictions, ignore_index=True)
    overall_metrics = pd.DataFrame(summaries)
    series_metrics = calculate_series_metrics(prediction_frame)
    _write_results(
        arguments.output_directory,
        frame,
        boundaries,
        split_indices,
        preprocessor,
        overall_metrics,
        series_metrics,
        prediction_frame,
        arguments.snapshot,
        evaluation_split,
    )
    print(overall_metrics.to_string(index=False))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="가격 예측 global model 성능 비교")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default=DEFAULT_EVALUATION_SPLIT,
    )
    return parser.parse_args()


def _load_snapshot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("가격 feature snapshot을 찾을 수 없습니다: " + str(path))
    frame = pd.read_csv(path, parse_dates=["price_date"])
    required = {"series_id", "price_date", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("모델 비교에 필요한 컬럼이 없습니다: " + ", ".join(missing))
    frame = frame.sort_values(["series_id", "price_date"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["series_id", "price_date"]).any():
        raise ValueError("모델 비교 snapshot에 중복된 시계열 날짜가 있습니다.")
    return frame


def _validate_split_coverage(frame: pd.DataFrame, split_indices: dict[str, np.ndarray]) -> None:
    expected_series = set(frame["series_id"].unique())
    for split_name, indices in split_indices.items():
        actual_series = set(frame.loc[indices, "series_id"].unique())
        if actual_series != expected_series:
            raise ValueError(f"{split_name} 구간에 포함되지 않은 가격 시계열이 있습니다.")


def _train_recurrent_models(frame, transformed, split_indices, evaluation_split):
    try:
        import tensorflow as tf
    except ImportError as exception:
        raise RuntimeError("LSTM/GRU 비교를 실행하려면 TensorFlow가 필요합니다.") from exception

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    tf.config.experimental.enable_op_determinism()

    x_train, y_train = build_sequences(
        transformed, frame, split_indices["train"], SEQUENCE_LENGTH, TARGET_COLUMN
    )
    x_validation, y_validation = build_sequences(
        transformed, frame, split_indices["validation"], SEQUENCE_LENGTH, TARGET_COLUMN
    )
    x_evaluation, _ = build_sequences(
        transformed, frame, split_indices[evaluation_split], SEQUENCE_LENGTH, TARGET_COLUMN
    )

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_validation_scaled = target_scaler.transform(y_validation.reshape(-1, 1)).ravel()

    results = []
    for model_name in ("lstm", "gru"):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_SEED)
        model = build_recurrent_model(model_name, x_train.shape[1:])
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        fit_start = time.perf_counter()
        history = model.fit(
            x_train,
            y_train_scaled,
            validation_data=(x_validation, y_validation_scaled),
            epochs=100,
            batch_size=32,
            shuffle=False,
            verbose=0,
            callbacks=[early_stopping],
        )
        fit_seconds = time.perf_counter() - fit_start
        prediction_start = time.perf_counter()
        prediction_scaled = model.predict(
            x_evaluation, batch_size=32, verbose=0
        ).reshape(-1, 1)
        predicted = target_scaler.inverse_transform(prediction_scaled).ravel()
        prediction_seconds = time.perf_counter() - prediction_start
        results.append(
            (model_name, predicted, fit_seconds, prediction_seconds, len(history.epoch))
        )
    return results


def _prediction_frame(model_name, frame, indices, predicted):
    return pd.DataFrame(
        {
            "model": model_name,
            "series_id": frame.loc[indices, "series_id"].to_numpy(),
            "price_date": frame.loc[indices, "price_date"].to_numpy(),
            "actual": frame.loc[indices, TARGET_COLUMN].to_numpy(dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
        }
    )


def _summary_row(
    model_name, actual, predicted, fit_seconds, prediction_seconds,
    split_indices, evaluation_split, epochs
):
    return {
        "model": model_name,
        **calculate_metrics(actual, predicted),
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "train_rows": len(split_indices["train"]),
        "validation_rows": len(split_indices["validation"]),
        "test_rows": len(split_indices["test"]),
        "evaluation_split": evaluation_split,
        "evaluation_rows": len(split_indices[evaluation_split]),
        "epochs": epochs,
    }


def _write_results(
    output_directory,
    frame,
    boundaries,
    split_indices,
    preprocessor,
    overall_metrics,
    series_metrics,
    predictions,
    snapshot_path,
    evaluation_split,
):
    output_directory.mkdir(parents=True, exist_ok=True)
    maximum_date = frame["price_date"].max().date().isoformat()
    overall_path = output_directory / f"{evaluation_split}_overall_metrics_as_of_{maximum_date}.csv"
    series_path = output_directory / f"{evaluation_split}_series_metrics_as_of_{maximum_date}.csv"
    predictions_path = output_directory / f"{evaluation_split}_predictions_as_of_{maximum_date}.csv"
    metadata_path = (
        output_directory
        / f"{evaluation_split}_model_comparison_as_of_{maximum_date}.metadata.json"
    )
    overall_metrics.to_csv(overall_path, index=False, encoding="utf-8")
    series_metrics.to_csv(series_path, index=False, encoding="utf-8")
    predictions.to_csv(predictions_path, index=False, date_format="%Y-%m-%d", encoding="utf-8")

    categorical_encoder = preprocessor.named_transformers_["categorical"]
    evaluation_keys = predictions[predictions["model"] == "baseline_lag_1"][
        ["series_id", "price_date"]
    ].to_csv(index=False)
    candidate_model = overall_metrics.sort_values(
        ["mae", "rmse"], kind="stable"
    ).iloc[0]["model"]
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(snapshot_path),
        "global_model": True,
        "series_identity_encoding": "train 기준 one-hot encoding",
        "series_categories": [int(value) for value in categorical_encoder.categories_[1]],
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_column": TARGET_COLUMN,
        "sequence_length": SEQUENCE_LENGTH,
        "evaluation_split": evaluation_split,
        "candidate_selection_metric": "validation MAE, 동률이면 RMSE",
        "candidate_model": candidate_model if evaluation_split == "validation" else None,
        "split": {
            "train_end": boundaries.train_end.date().isoformat(),
            "validation_end": boundaries.validation_end.date().isoformat(),
            "train_rows": len(split_indices["train"]),
            "validation_rows": len(split_indices["validation"]),
            "test_rows": len(split_indices["test"]),
        },
        "evaluation_key_sha256": hashlib.sha256(
            evaluation_keys.encode("utf-8")
        ).hexdigest(),
        "models": overall_metrics["model"].tolist(),
        "model_artifacts_saved": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
