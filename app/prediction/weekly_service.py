from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd

from app.prediction.feature_builder import (
    build_latest_weekly_feature,
)
from app.prediction.model_bundle import (
    WeeklyPriceModelBundle,
)
from app.schemas.price_prediction import (
    WeeklyPricePredictionBatchRequest,
    WeeklyPricePredictionResponse,
    WeeklyPriceSeriesRequest,
)


# 현재 주간 모델은 기준일 이후 7일 구간을 예측 대상으로 한다.
SEVEN_DAY_HORIZON = 7


class UnsupportedPredictionSeriesError(ValueError):
    """
    현재 운영 artifact가 학습하지 않은 series가
    예측 요청에 포함된 경우 발생한다.
    """


class WeeklyPricePredictionService:
    """
    주간 가격예측의 애플리케이션 서비스.

    전체 흐름:
    1. Backend가 DB에서 가격 이력을 조회한다.
    2. Backend가 서울·부산·대전의 일별 대표가격을 전달한다.
    3. AI Server가 가격 이력을 모델 feature로 변환한다.
    4. Ridge 모델로 다음 7일 평균/최대 가격을 예측한다.
    5. ML 기반 위험 점수를 계산한다.
    6. 예측 결과를 Backend에 반환한다.

    주의:
    - DB 조회/저장은 Backend 책임이다.
    - feature 계산 및 ML 추론은 AI Server 책임이다.
    - 최종 위험등급 판정은 Backend 비즈니스 로직에서 처리한다.
    """

    def __init__(
        self,
        model_bundle: WeeklyPriceModelBundle | None = None,
    ) -> None:
        """
        운영 모델 artifact를 준비한다.

        테스트에서는 외부에서 model_bundle을 주입할 수 있고,
        실제 실행에서는 기본 weekly_ridge_v1 artifact를 로드한다.
        """

        self._model_bundle = (
            model_bundle
            if model_bundle is not None
            else WeeklyPriceModelBundle()
        )

    def predict(
        self,
        request: WeeklyPricePredictionBatchRequest,
    ) -> list[WeeklyPricePredictionResponse]:
        """
        하나의 요청에 포함된 모든 series를 순서대로 예측한다.
        """

        # 먼저 요청된 series가 현재 운영 모델의
        # 학습 대상인지 확인한다.
        #
        # 학습하지 않은 series를 OneHotEncoder의 unknown 처리만으로
        # 그대로 예측하는 것은 운영상 신뢰하기 어렵기 때문에 차단한다.
        self._validate_supported_series(request)

        # 같은 batch 요청에서 생성되는 결과들은
        # 동일한 generatedAt 값을 사용한다.
        generated_at = datetime.now(UTC)

        # 각 series마다 독립적으로 feature 생성 및 모델 추론 수행
        return [
            self._predict_one(
                series=series,
                generated_at=generated_at,
            )
            for series in request.series
        ]

    def _predict_one(
        self,
        series: WeeklyPriceSeriesRequest,
        generated_at: datetime,
    ) -> WeeklyPricePredictionResponse:
        """
        하나의 canonical series에 대해
        실제 주간 가격예측을 수행한다.
        """

        # API 요청으로 받은 가격 배열을
        # feature_builder가 사용하는 DataFrame으로 변환한다.
        price_frame = self._to_price_frame(
            series
        )

        # 학습 당시와 동일한 계산 규칙을 적용해
        # 가장 최근 날짜 기준 feature 1행을 생성한다.
        #
        # 여기서 lag, rolling mean/std, return,
        # trend, season 등의 값이 계산된다.
        feature = build_latest_weekly_feature(
            price_frame
        )

        # 저장된 preprocessor와 Ridge 모델을 사용해
        # 평균가격, 최대가격, 위험 점수를 계산한다.
        predictions = self._model_bundle.predict(
            feature
        )

        # 하나의 series로 feature 1행을 만들었으므로
        # 모델 결과 역시 반드시 1건이어야 한다.
        if len(predictions) != 1:
            raise RuntimeError(
                "주간 가격예측 결과가 정확히 1건이어야 합니다."
            )

        prediction = predictions[0]

        # feature의 기준일이 실제 예측 기준일이다.
        base_date = pd.Timestamp(
            feature.iloc[0]["base_date"]
        ).date()

        # feature 계산은 float를 사용하지만 응답의 기준가격은
        # Backend가 전달한 Decimal을 그대로 유지해 검증·저장 정밀도를 보장한다.
        base_price = next(
            point.representative_price
            for point in series.prices
            if point.price_date == base_date
        )

        # 다음 7일 평균 예상가격.
        #
        # 현재 price_prediction 테이블의
        # predicted_price 컬럼에는 이 값만 저장한다.
        predicted_price = Decimal(
            str(
                prediction.predicted_mean_price
            )
        )

        # 다음 7일 동안의 최대 예상가격.
        #
        # 가격 위험 신호 계산에는 사용하지만
        # 현재 price_prediction 테이블에는 저장하지 않는다.
        predicted_max_price = Decimal(
            str(
                prediction.predicted_max_price
            )
        )

        return WeeklyPricePredictionResponse(
            series_id=series.series_id,

            # 예측 기준일
            base_date=base_date,

            # 주간 예측 구간의 마지막 날짜
            target_date=(
                base_date
                + timedelta(
                    days=SEVEN_DAY_HORIZON
                )
            ),

            # 현재 대표가격
            base_price=base_price,

            # 다음 7일 평균 예상가격
            predicted_price=predicted_price,

            # 다음 7일 최대 예상가격
            predicted_max_price=(
                predicted_max_price
            ),

            # 가격 단위는 Backend가 전달한 값을 유지한다.
            standard_unit=series.standard_unit,

            # max Ridge 예상 변화율의
            # 학습 분포 percentile
            ridge_score=prediction.ridge_score,

            # 최근 가격 변동성의
            # 학습 분포 percentile
            volatility_score=(
                prediction.volatility_score
            ),

            # 두 점수를 동일 가중치로 결합한
            # ML 기반 종합 위험 점수
            combined_risk_score=(
                prediction.combined_risk_score
            ),

            # manifest에 저장된 실제 모델 정보
            model_name=(
                self._model_bundle.model_name
            ),
            model_version=(
                self._model_bundle.model_version
            ),

            # 해당 batch의 예측 생성 시각
            generated_at=generated_at,
        )

    @staticmethod
    def _to_price_frame(
        series: WeeklyPriceSeriesRequest,
    ) -> pd.DataFrame:
        """
        Backend에서 전달받은 가격 이력을
        feature_builder 입력 형식으로 변환한다.

        이 단계에서는 feature를 계산하지 않는다.
        단순한 데이터 구조 변환만 수행한다.
        """

        rows = [
            {
                # 학습 데이터에서 사용한 canonical series ID
                "series_id": series.series_id,

                # categorical feature로 사용되는 식재료 코드
                "ingredient_code": (
                    series.ingredient_code
                ),

                # 해당 대표가격의 날짜
                "price_date": (
                    point.price_date
                ),

                # 서울·부산·대전 3지역의
                # 해당 날짜 표준단가 평균
                "representative_standard_unit_price": (
                    point.representative_price
                ),
            }
            for point in series.prices
        ]

        return pd.DataFrame(rows)

    def _validate_supported_series(
        self,
        request: WeeklyPricePredictionBatchRequest,
    ) -> None:
        """
        요청 series가 현재 모델 학습 대상인지 확인한다.

        weekly_ridge_v1은 학습 당시 검증된
        44개의 canonical series만 운영 대상으로 사용한다.
        """

        supported = (
            self._model_bundle
            .supported_series_ids
        )

        # 요청 ID 중 artifact manifest에 없는 ID만 추출
        unsupported = sorted(
            series.series_id
            for series in request.series
            if series.series_id
            not in supported
        )

        # 지원하지 않는 ID가 하나라도 있으면
        # 일부만 예측하지 않고 요청 전체를 실패시킨다.
        if unsupported:
            joined = ", ".join(
                str(series_id)
                for series_id in unsupported
            )

            raise UnsupportedPredictionSeriesError(
                "현재 주간 예측 모델이 지원하지 않는 "
                f"시계열입니다: {joined}"
            )
