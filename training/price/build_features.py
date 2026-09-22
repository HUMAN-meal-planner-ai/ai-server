from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from training.price.config import DatabaseSettings
from training.price.database import open_read_only_connection
from training.price.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_price_features,
)
from training.price.price_repository import PriceDataRepository


DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "snapshots"


def main() -> None:
    arguments = _parse_arguments()
    settings = DatabaseSettings.from_environment(arguments.env_file)

    with open_read_only_connection(settings) as connection:
        repository = PriceDataRepository(connection)
        eligibility = repository.load_series_eligibility()
        prices = repository.load_eligible_prices(arguments.minimum_observations)

    features = build_price_features(prices)
    csv_path, metadata_path = _write_snapshot(
        features,
        eligibility,
        arguments.output_directory,
        arguments.minimum_observations,
    )
    print(
        "가격 feature snapshot 생성을 완료했습니다. "
        f"대상 시계열={features['series_id'].nunique()}, 행={len(features)}, "
        f"CSV={csv_path}, 메타데이터={metadata_path}"
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KAMIS 가격 시계열 feature snapshot 생성")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--minimum-observations", type=int, default=200)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def _write_snapshot(features, eligibility, output_directory: Path, minimum_observations: int):
    output_directory.mkdir(parents=True, exist_ok=True)
    maximum_date = features["price_date"].max().date().isoformat()
    csv_path = output_directory / f"price_features_as_of_{maximum_date}.csv"
    metadata_path = output_directory / f"price_features_as_of_{maximum_date}.metadata.json"

    features.to_csv(csv_path, index=False, date_format="%Y-%m-%d", encoding="utf-8")

    excluded = eligibility[eligibility["observation_count"] < minimum_observations].copy()
    exclusions = []
    for row in excluded.itertuples(index=False):
        count = int(row.observation_count)
        exclusions.append(
            {
                "series_id": int(row.series_id),
                "ingredient_code": row.ingredient_code,
                "ingredient_name": row.ingredient_name,
                "observation_count": count,
                "reason": "가격 데이터 없음" if count == 0 else "200건 미만",
            }
        )

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_tables": ["mealfit.price_series", "mealfit.ingredient_price"],
        "source_name": "KAMIS",
        "minimum_observations": minimum_observations,
        "lag_semantics": "series별 관측치 기준",
        "target_column": TARGET_COLUMN,
        "target_unit": "원/g",
        "feature_columns": list(FEATURE_COLUMNS),
        "audit_columns": ["original_price"],
        "eligible_series_count": int(features["series_id"].nunique()),
        "feature_row_count": int(len(features)),
        "first_feature_date": features["price_date"].min().date().isoformat(),
        "last_feature_date": maximum_date,
        "excluded_series": exclusions,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return csv_path, metadata_path


if __name__ == "__main__":
    main()
