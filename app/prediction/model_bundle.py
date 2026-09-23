from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# 운영용 가격예측 artifact 기본 경로
DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "price_model"
    / "weekly_ridge_v1"
)


@dataclass(frozen=True)
class WeeklyPrediction:
    """주간 가격예측 모델의 추론 결과."""

    # 다음 7일 평균 예상가격
    predicted_mean_price: float

    # 다음 7일 내 최대 예상가격
    predicted_max_price: float

    # max Ridge 예측 변화율의 경험적 percentile 점수
    ridge_score: float

    # 최근 7일 변동성의 경험적 percentile 점수
    volatility_score: float

    # Ridge 점수와 변동성 점수를 동일 가중 평균한 위험 점수
    combined_risk_score: float


class WeeklyPriceModelBundle:
    """
    주간 가격예측에 필요한 artifact를 한 번에 로드하고 추론한다.

    구성:
    - 전처리기
    - 7일 평균가격 Ridge 모델
    - 7일 최대가격 Ridge 모델
    - 위험 점수 계산용 calibration 분포
    - 모델 메타데이터 manifest
    """

    def __init__(
        self,
        artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    ) -> None:
        self._artifact_dir = artifact_dir

        # 학습 시 사용한 동일 전처리기 로드
        self._preprocessor = joblib.load(
            artifact_dir / "preprocessor.joblib"
        )

        # 다음 7일 평균가격 예측 모델
        self._mean_model = joblib.load(
            artifact_dir / "mean_ridge.joblib"
        )

        # 다음 7일 최대가격 예측 모델
        self._max_model = joblib.load(
            artifact_dir / "max_ridge.joblib"
        )

        # 위험 점수를 percentile로 변환하기 위한 학습 기준 분포
        calibration = np.load(
            artifact_dir / "risk_calibration.npz"
        )

        self._ridge_reference = np.asarray(
            calibration["ridge_return_reference"],
            dtype=float,
        )

        self._volatility_reference = np.asarray(
            calibration["volatility_reference"],
            dtype=float,
        )

        # 모델 버전, threshold, 지원 series 등 메타데이터
        self._manifest = json.loads(
            (artifact_dir / "manifest.json").read_text(
                encoding="utf-8"
            )
        )

    @property
    def risk_threshold(self) -> float:
        """운영 위험 판정 기준값을 반환한다."""
        return float(self._manifest["risk_threshold"])

    @property
    def supported_series_ids(self) -> set[int]:
        """현재 artifact가 학습한 series ID 목록을 반환한다."""
        return {
            int(value)
            for value in self._manifest["supported_series_ids"]
        }

    def predict(
        self,
        features: pd.DataFrame,
    ) -> list[WeeklyPrediction]:
        """
        생성된 feature DataFrame으로 주간 가격과 위험 점수를 예측한다.

        Python에서는 위험 점수까지만 계산하고,
        실제 위험등급 판정은 Backend(Java) 비즈니스 로직에서 처리한다.
        """
        if features.empty:
            return []

        # 학습 시와 동일한 전처리 적용
        transformed = np.asarray(
            self._preprocessor.transform(features),
            dtype=np.float32,
        )

        # 다음 7일 평균가격과 최대가격 각각 예측
        predicted_mean = self._mean_model.predict(transformed)
        predicted_max = self._max_model.predict(transformed)

        current_price = features["current_price"].to_numpy(
            dtype=float
        )

        # 최근 7일 표준편차를 현재가격으로 나눠
        # 가격 규모와 무관한 상대 변동성으로 변환
        volatility = (
            features["std_7d"].to_numpy(dtype=float)
            / current_price
        )

        # 최대 예상가격이 현재가격 대비 얼마나 상승하는지 계산
        predicted_max_return = (
            predicted_max / current_price - 1
        )

        # Ridge 예측 변화율을 학습 분포 기준 percentile 점수로 변환
        ridge_score = self._percentile(
            self._ridge_reference,
            predicted_max_return,
        )

        # 현재 변동성도 동일하게 percentile 점수로 변환
        volatility_score = self._percentile(
            self._volatility_reference,
            volatility,
        )

        # 검증에서 사용한 방식과 동일하게 두 점수를 1:1로 결합
        combined_score = (
            ridge_score + volatility_score
        ) / 2

        return [
            WeeklyPrediction(
                predicted_mean_price=float(mean_price),
                predicted_max_price=float(max_price),
                ridge_score=float(ridge),
                volatility_score=float(volatility_value),
                combined_risk_score=float(combined),
            )
            for (
                mean_price,
                max_price,
                ridge,
                volatility_value,
                combined,
            ) in zip(
                predicted_mean,
                predicted_max,
                ridge_score,
                volatility_score,
                combined_score,
                strict=True,
            )
        ]

    @staticmethod
    def _percentile(
        reference: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        입력값을 학습 데이터의 기준 분포와 비교해
        0~1 범위의 경험적 percentile 점수로 변환한다.

        예:
        0.80이면 해당 값이 학습 기준 분포의
        약 80%보다 높은 위치에 있다는 의미이다.
        """

        # 기준 분포가 비어 있으면 percentile을 계산할 수 없으므로
        # 모델 추론 자체를 중단한다.
        if reference.size == 0:
            raise ValueError(
                "위험 점수 계산용 기준 분포가 비어 있습니다."
            )

        # reference는 artifact 생성 시 오름차순으로 저장되어 있다.
        #
        # searchsorted(..., side="right")는
        # 현재 값 이하에 해당하는 기준값 개수를 반환한다.
        #
        # 이를 전체 기준값 개수로 나누면
        # 0~1 사이의 경험적 percentile 값이 된다.
        return np.searchsorted(
            reference,
            values,
            side="right",
        ) / reference.size

    @property
    def model_name(self) -> str:
        """
        현재 운영 중인 주간 평균가격 예측 모델명을 반환한다.

        모델명은 코드에 직접 하드코딩하지 않고
        artifact 생성 당시 manifest.json에 기록된 값을 사용한다.
        """
        return str(
            self._manifest["mean_model_name"]
        )

    @property
    def model_version(self) -> str:
        """
        현재 로드된 운영 모델 artifact의 버전을 반환한다.

        Backend에 예측 결과를 전달할 때 함께 반환하여
        어떤 모델 버전으로 생성된 예측인지 추적할 수 있게 한다.
        """
        return str(
            self._manifest["model_version"]
        )