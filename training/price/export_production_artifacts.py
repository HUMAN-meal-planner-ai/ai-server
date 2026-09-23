from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from training.price.analyze_weekly_forecast import (
    MAX_TARGET_COLUMN,
    TARGET_COLUMN,
)
from training.price.compare_regional_weekly_models import (
    CATEGORICAL_FEATURES,
    DEFAULT_SNAPSHOT,
    NUMERIC_FEATURES,
    build_preprocessor,
    load_snapshot,
)

ARTIFACT_VERSION = "weekly_ridge_v1"
RISK_THRESHOLD = 0.75

DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "price_model"
    / ARTIFACT_VERSION
)


def main() -> None:
    frame = load_snapshot(DEFAULT_SNAPSHOT)

    preprocessor = build_preprocessor()
    preprocessor.fit(frame)

    x_train = np.asarray(
        preprocessor.transform(frame),
        dtype=np.float32,
    )

    mean_model = Ridge(alpha=1.0)
    mean_model.fit(
        x_train,
        frame[TARGET_COLUMN].to_numpy(dtype=float),
    )

    max_model = Ridge(alpha=1.0)
    max_model.fit(
        x_train,
        frame[MAX_TARGET_COLUMN].to_numpy(dtype=float),
    )

    current_price = frame["current_price"].to_numpy(dtype=float)

    max_predicted_price = max_model.predict(x_train)
    ridge_return_reference = (
        max_predicted_price / current_price - 1
    )

    volatility_reference = (
        frame["std_7d"].to_numpy(dtype=float)
        / current_price
    )

    DEFAULT_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        preprocessor,
        DEFAULT_ARTIFACT_DIR / "preprocessor.joblib",
    )

    joblib.dump(
        mean_model,
        DEFAULT_ARTIFACT_DIR / "mean_ridge.joblib",
    )

    joblib.dump(
        max_model,
        DEFAULT_ARTIFACT_DIR / "max_ridge.joblib",
    )

    np.savez(
        DEFAULT_ARTIFACT_DIR / "risk_calibration.npz",
        ridge_return_reference=np.sort(
            ridge_return_reference
        ),
        volatility_reference=np.sort(
            volatility_reference
        ),
    )

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(DEFAULT_SNAPSHOT),
        "training_rows": len(frame),
        "training_start_date": (
            frame["base_date"].min().date().isoformat()
        ),
        "training_end_date": (
            frame["base_date"].max().date().isoformat()
        ),
        "supported_series_ids": sorted(
            int(value)
            for value in frame["series_id"].unique()
        ),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(
            CATEGORICAL_FEATURES
        ),
        "mean_target": TARGET_COLUMN,
        "max_target": MAX_TARGET_COLUMN,
        "mean_model_name": "weekly_mean_ridge",
        "max_model_name": "weekly_max_ridge",
        "model_version": ARTIFACT_VERSION,
        "risk_threshold": RISK_THRESHOLD,
        "risk_score": (
            "mean(percentile(max_ridge_return), "
            "percentile(std_7d/current_price))"
        ),
    }

    (
        DEFAULT_ARTIFACT_DIR / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"artifact 생성 완료: {DEFAULT_ARTIFACT_DIR}"
    )


if __name__ == "__main__":
    main()