from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from training.price.compare_models import (
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_SNAPSHOT,
    RANDOM_SEED,
    _load_snapshot,
)
from training.price.evaluation_metrics import calculate_metrics
from training.price.feature_engineering import TARGET_COLUMN
from training.price.model_definitions import build_preprocessor, build_recurrent_model
from training.price.sequence_builder import build_sequences, sequence_eligible_indices
from training.price.time_split import determine_time_boundaries, split_name_for_dates


SEQUENCE_LENGTHS = (7, 14, 30)
HIDDEN_SIZES = (16, 32, 64)
EVALUATION_SPLIT = "validation"


def main() -> None:
    frame = _load_snapshot(DEFAULT_SNAPSHOT)
    boundaries = determine_time_boundaries(frame)
    frame["split"] = split_name_for_dates(frame["price_date"], boundaries)

    preprocessor = build_preprocessor()
    preprocessor.fit(frame[frame["split"] == "train"])
    transformed = np.asarray(preprocessor.transform(frame), dtype=np.float32)

    validation_indices = frame.index[frame["split"] == EVALUATION_SPLIT].to_numpy(dtype=int)
    actual_validation = frame.loc[validation_indices, TARGET_COLUMN].to_numpy(dtype=float)
    baseline_prediction = frame.loc[validation_indices, "lag_1"].to_numpy(dtype=float)
    results = [
        {
            "model": "baseline_lag_1",
            "sequence_length": None,
            "hidden_size": None,
            **calculate_metrics(actual_validation, baseline_prediction),
            "fit_seconds": 0.0,
            "prediction_seconds": 0.0,
            "epochs": None,
        }
    ]
    sequence_counts = []

    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    for sequence_length in SEQUENCE_LENGTHS:
        eligible_indices = sequence_eligible_indices(frame, sequence_length)
        split_indices = {
            split_name: np.asarray(
                [
                    index
                    for index in eligible_indices
                    if frame.at[index, "split"] == split_name
                ],
                dtype=int,
            )
            for split_name in ("train", "validation", "test")
        }
        _record_sequence_counts(frame, split_indices, sequence_length, sequence_counts)
        if not np.array_equal(split_indices[EVALUATION_SPLIT], validation_indices):
            raise ValueError("sequence length별 Validation 평가 행이 서로 다릅니다.")

        x_train, y_train = build_sequences(
            transformed,
            frame,
            split_indices["train"],
            sequence_length,
            TARGET_COLUMN,
        )
        x_validation, y_validation = build_sequences(
            transformed,
            frame,
            split_indices[EVALUATION_SPLIT],
            sequence_length,
            TARGET_COLUMN,
        )
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_validation_scaled = target_scaler.transform(
            y_validation.reshape(-1, 1)
        ).ravel()

        for model_name in ("lstm", "gru"):
            for hidden_size in HIDDEN_SIZES:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(RANDOM_SEED)
                model = build_recurrent_model(
                    model_name, x_train.shape[1:], hidden_size=hidden_size
                )
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
                    x_validation, batch_size=32, verbose=0
                ).reshape(-1, 1)
                predicted = target_scaler.inverse_transform(prediction_scaled).ravel()
                prediction_seconds = time.perf_counter() - prediction_start
                metrics = calculate_metrics(actual_validation, predicted)
                results.append(
                    {
                        "model": model_name,
                        "sequence_length": sequence_length,
                        "hidden_size": hidden_size,
                        **metrics,
                        "fit_seconds": fit_seconds,
                        "prediction_seconds": prediction_seconds,
                        "epochs": len(history.epoch),
                    }
                )
                print(
                    f"완료: model={model_name}, sequence={sequence_length}, "
                    f"hidden={hidden_size}, MAE={metrics['mae']:.6f}, "
                    f"epochs={len(history.epoch)}",
                    flush=True,
                )

    _write_diagnostics(
        pd.DataFrame(results),
        pd.DataFrame(sequence_counts),
        frame,
        boundaries,
    )


def _record_sequence_counts(frame, split_indices, sequence_length, output) -> None:
    for split_name, indices in split_indices.items():
        counts = frame.loc[indices].groupby("series_id").size()
        for series_id in sorted(frame["series_id"].unique()):
            output.append(
                {
                    "sequence_length": sequence_length,
                    "split": split_name,
                    "series_id": int(series_id),
                    "sequence_count": int(counts.get(series_id, 0)),
                }
            )


def _write_diagnostics(results, sequence_counts, frame, boundaries) -> None:
    output_directory = Path(DEFAULT_OUTPUT_DIRECTORY)
    output_directory.mkdir(parents=True, exist_ok=True)
    maximum_date = frame["price_date"].max().date().isoformat()
    results_path = output_directory / f"dl_config_metrics_as_of_{maximum_date}.csv"
    counts_path = output_directory / f"dl_sequence_counts_as_of_{maximum_date}.csv"
    metadata_path = output_directory / f"dl_diagnostics_as_of_{maximum_date}.metadata.json"
    results.to_csv(results_path, index=False, encoding="utf-8")
    sequence_counts.to_csv(counts_path, index=False, encoding="utf-8")

    dl_results = results[results["model"].isin(["lstm", "gru"])]
    best_dl = dl_results.sort_values(["mae", "rmse"], kind="stable").iloc[0]
    baseline = results[results["model"] == "baseline_lag_1"].iloc[0]
    train_counts = sequence_counts[sequence_counts["split"] == "train"]
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(DEFAULT_SNAPSHOT),
        "evaluation_split": EVALUATION_SPLIT,
        "train_end": boundaries.train_end.date().isoformat(),
        "validation_end": boundaries.validation_end.date().isoformat(),
        "validation_rows": int((frame["split"] == EVALUATION_SPLIT).sum()),
        "sequence_lengths": list(SEQUENCE_LENGTHS),
        "hidden_sizes": list(HIDDEN_SIZES),
        "best_dl": {
            "model": best_dl["model"],
            "sequence_length": int(best_dl["sequence_length"]),
            "hidden_size": int(best_dl["hidden_size"]),
            "mae": float(best_dl["mae"]),
        },
        "baseline_mae": float(baseline["mae"]),
        "best_dl_beats_baseline": bool(best_dl["mae"] < baseline["mae"]),
        "train_sequence_count_range": {
            str(length): {
                "minimum": int(group["sequence_count"].min()),
                "maximum": int(group["sequence_count"].max()),
                "mean": float(group["sequence_count"].mean()),
            }
            for length, group in train_counts.groupby("sequence_length")
        },
        "model_artifacts_saved": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(results.sort_values(["mae", "rmse"], kind="stable").to_string(index=False))


if __name__ == "__main__":
    main()
