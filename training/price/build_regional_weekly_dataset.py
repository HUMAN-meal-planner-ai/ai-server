from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from training.price.analyze_weekly_forecast import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_weekly_forecast_frame,
)
from training.price.config import DatabaseSettings
from training.price.database import open_read_only_connection
from training.price.price_repository import PriceDataRepository


MINIMUM_COMPLETE_OBSERVATIONS = 200
EXCLUDED_SOURCE_SERIES_IDS = (4, 50, 59)
REPRESENTATIVE_PRICE_COLUMN = "representative_standard_unit_price"
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "snapshots"


def build_regional_weekly_dataset(representative_prices: pd.DataFrame) -> pd.DataFrame:
    """3지역 일별 대표가격에서 기준일 이후 7일 평균을 target으로 만든다."""
    required = {
        "series_id",
        "ingredient_code",
        "price_date",
        REPRESENTATIVE_PRICE_COLUMN,
    }
    missing = sorted(required.difference(representative_prices.columns))
    if missing:
        raise ValueError("지역 대표가격 데이터에 필요한 컬럼이 없습니다: " + ", ".join(missing))

    source = representative_prices.copy()
    source["price_date"] = pd.to_datetime(source["price_date"], errors="raise")
    source[REPRESENTATIVE_PRICE_COLUMN] = pd.to_numeric(
        source[REPRESENTATIVE_PRICE_COLUMN], errors="raise"
    )
    source = source.sort_values(["series_id", "price_date"], kind="stable")
    if source.duplicated(["series_id", "price_date"]).any():
        raise ValueError("지역 대표가격에 중복된 시계열 날짜가 있습니다.")

    weekly_source = source.rename(
        columns={REPRESENTATIVE_PRICE_COLUMN: "standard_unit_price"}
    )
    dataset = build_weekly_forecast_frame(weekly_source)
    dataset = dataset.merge(
        source[["series_id", "price_date", "ingredient_code"]],
        left_on=["series_id", "base_date"],
        right_on=["series_id", "price_date"],
        how="left",
        validate="one_to_one",
    ).drop(columns="price_date")
    return dataset.sort_values(["series_id", "base_date"], kind="stable").reset_index(
        drop=True
    )


def main() -> None:
    arguments = _parse_arguments()
    settings = DatabaseSettings.from_environment(arguments.env_file)
    with open_read_only_connection(settings) as connection:
        prices = PriceDataRepository(connection).load_regional_representative_prices(
            minimum_observations=arguments.minimum_observations,
            excluded_series_ids=EXCLUDED_SOURCE_SERIES_IDS,
        )

    dataset = build_regional_weekly_dataset(prices)
    csv_path, metadata_path = _write_snapshot(
        dataset,
        prices,
        arguments.output_directory,
        arguments.minimum_observations,
    )
    print(
        "3지역 대표가격 주간 학습 데이터셋 생성을 완료했습니다. "
        f"대상 series={dataset['series_id'].nunique()}, "
        f"ingredient={dataset['ingredient_code'].nunique()}, 행={len(dataset)}, "
        f"CSV={csv_path}, 메타데이터={metadata_path}"
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3지역 대표가격 주간 학습 데이터셋 생성")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--minimum-observations", type=int, default=MINIMUM_COMPLETE_OBSERVATIONS)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def _write_snapshot(
    dataset: pd.DataFrame,
    representative_prices: pd.DataFrame,
    output_directory: Path,
    minimum_observations: int,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    maximum_date = dataset["base_date"].max().date().isoformat()
    csv_path = output_directory / f"regional_weekly_features_as_of_{maximum_date}.csv"
    metadata_path = output_directory / f"regional_weekly_features_as_of_{maximum_date}.metadata.json"
    dataset.to_csv(csv_path, index=False, date_format="%Y-%m-%d", encoding="utf-8")

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_tables": ["mealfit.price_series", "mealfit.ingredient_price"],
        "representative_price": "서울·부산·대전이 모두 있는 일자별 AVG(standard_unit_price)",
        "included_regions": ["서울", "부산", "대전"],
        "excluded_source_series_ids": list(EXCLUDED_SOURCE_SERIES_IDS),
        "minimum_complete_observations": minimum_observations,
        "target_column": TARGET_COLUMN,
        "target_definition": "기준일 다음날부터 7일 이내 실제 공시 대표가격의 평균; 최소 3개 관측일 필요",
        "numeric_feature_columns": list(NUMERIC_FEATURES),
        "categorical_feature_columns": [*CATEGORICAL_FEATURES, "ingredient_code"],
        "source_series_count": int(representative_prices["series_id"].nunique()),
        "ingredient_count": int(representative_prices["ingredient_code"].nunique()),
        "representative_price_row_count": int(len(representative_prices)),
        "dataset_row_count": int(len(dataset)),
        "first_base_date": dataset["base_date"].min().date().isoformat(),
        "last_base_date": maximum_date,
        "first_target_end_date": dataset["target_end_date"].min().date().isoformat(),
        "last_target_end_date": dataset["target_end_date"].max().date().isoformat(),
        "series_sample_counts": {
            str(series_id): int(count)
            for series_id, count in dataset.groupby("series_id").size().items()
        },
        "database_written": False,
        "model_training_executed": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return csv_path, metadata_path


if __name__ == "__main__":
    main()
