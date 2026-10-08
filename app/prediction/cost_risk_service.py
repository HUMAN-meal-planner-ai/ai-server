from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from app.schemas.cost_risk import (
    MenuRiskFeatures,
    MenuRiskPrediction,
    MealPlanRiskFeatures,
    MealPlanRiskPrediction,
)

# 기본 모델 아티팩트 경로
DEFAULT_COST_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts" / "cost_model"
)


class CostRiskModelBundle:
    """
    메뉴 및 식단 종합 위험도 분류 LightGBM 모델 로더 및 추론 엔진.
    """

    def __init__(self, artifact_dir: Path = DEFAULT_COST_ARTIFACT_DIR) -> None:
        self._artifact_dir = artifact_dir

        if not artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

        # 메뉴 위험도 모델 로드
        self._menu_model = joblib.load(artifact_dir / "menu_risk_lgbm.pkl")

        # 식단 종합 위험도 모델 로드
        self._mealplan_model = joblib.load(artifact_dir / "mealplan_risk_lgbm.pkl")

        # 메타데이터 로드
        metadata_file = artifact_dir / "metadata.json"
        if metadata_file.exists():
            self._metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        else:
            self._metadata = {}

        self._menu_classes = {0: "SAFE", 1: "CAUTION", 2: "WARNING"}
        self._mealplan_classes = {0: "SAFE", 1: "CAUTION", 2: "WARNING", 3: "CRITICAL"}

        # 메타데이터에 등록된 피처 목록 추출
        self._menu_feature_names = self._metadata.get("menu_risk_model", {}).get(
            "features",
            [
                "category_code",
                "ingredient_count",
                "cur_menu_cost",
                "menu_cost_increase_rate",
                "max_ingredient_increase_rate",
                "max_contribution_rate",
                "top_driver_risk",
                "high_risk_ingredient_count",
                "caution_ingredient_count",
                "avg_price_volatility",
            ],
        )

        self._mealplan_feature_names = self._metadata.get("mealplan_risk_model", {}).get(
            "features",
            [
                "total_menu_count",
                "warning_menu_ratio",
                "caution_menu_ratio",
                "risk_cost_ratio",
                "warning_cost_ratio",
                "total_expected_cost",
                "budget_usage_ratio",
                "avg_menu_cost_increase_rate",
                "max_menu_cost_increase_rate",
            ],
        )

    @property
    def metadata(self) -> dict:
        return self._metadata

    def predict_menu_risks(
        self, items: List[MenuRiskFeatures]
    ) -> List[MenuRiskPrediction]:
        """메뉴 위험도 배치 예측 수행."""
        if not items:
            return []

        # DataFrame으로 변환
        records = [item.model_dump() for item in items]
        df = pd.DataFrame(records)

        # 모델 입력에 필요한 피처만 순서대로 추출
        X = df[self._menu_feature_names]

        preds = self._menu_model.predict(X)
        probas = self._menu_model.predict_proba(X)

        results = []
        for i, item in enumerate(items):
            code = int(preds[i])
            prob_arr = probas[i]
            level = self._menu_classes.get(code, "UNKNOWN")
            confidence = float(np.max(prob_arr))

            # 100점 만점 위험 점수 환산: SAFE(0~33), CAUTION(34~66), WARNING(67~100)
            # 확률 가중치를 적용한 연속형 스코어 계산
            # prob[0]*15 + prob[1]*50 + prob[2]*90
            risk_score = (
                float(prob_arr[0]) * 15.0
                + (float(prob_arr[1]) * 55.0 if len(prob_arr) > 1 else 0.0)
                + (float(prob_arr[2]) * 90.0 if len(prob_arr) > 2 else 0.0)
            )
            risk_score = round(min(100.0, max(0.0, risk_score)), 1)

            prob_dict = {
                self._menu_classes.get(c_idx, str(c_idx)): round(float(p), 4)
                for c_idx, p in enumerate(prob_arr)
            }

            results.append(
                MenuRiskPrediction(
                    menu_id=item.menu_id,
                    risk_level=level,
                    risk_code=code,
                    risk_score=risk_score,
                    confidence=round(confidence, 4),
                    probabilities=prob_dict,
                )
            )

        return results

    def predict_mealplan_risks(
        self, items: List[MealPlanRiskFeatures]
    ) -> List[MealPlanRiskPrediction]:
        """식단 종합 위험도 배치 예측 수행."""
        if not items:
            return []

        results = []
        to_predict_indices = []
        to_predict_features = []

        # 1. 빈 식단(total_menu_count == 0)은 추론 없이 즉시 SAFE로 처리 (IT-D03 대응)
        for i, item in enumerate(items):
            if item.total_menu_count <= 0 or item.total_expected_cost <= 0:
                results.append(
                    MealPlanRiskPrediction(
                        plan_id=item.plan_id,
                        facility_id=item.facility_id,
                        risk_level="SAFE",
                        risk_code=0,
                        risk_score=0.0,
                        confidence=1.0,
                        probabilities={"SAFE": 1.0, "CAUTION": 0.0, "WARNING": 0.0, "CRITICAL": 0.0},
                    )
                )
            else:
                to_predict_indices.append(i)
                to_predict_features.append(item.model_dump())
                results.append(None)  # Placeholder

        if not to_predict_indices:
            return results

        # 2. 유효한 식단에 대해서만 모델 추론 수행
        df = pd.DataFrame(to_predict_features)
        for col in self._mealplan_feature_names:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self._mealplan_feature_names].fillna(0.0)

        preds = self._mealplan_model.predict(X)
        probas = self._mealplan_model.predict_proba(X)

        for p_idx, orig_idx in enumerate(to_predict_indices):
            item = items[orig_idx]
            code = int(preds[p_idx])
            prob_arr = probas[p_idx]
            level = self._mealplan_classes.get(code, "UNKNOWN")
            confidence = float(np.max(prob_arr))

            # 100점 만점 위험 점수 환산: SAFE(0~25), CAUTION(26~50), WARNING(51~75), CRITICAL(76~100)
            risk_score = (
                float(prob_arr[0]) * 12.0
                + (float(prob_arr[1]) * 38.0 if len(prob_arr) > 1 else 0.0)
                + (float(prob_arr[2]) * 68.0 if len(prob_arr) > 2 else 0.0)
                + (float(prob_arr[3]) * 95.0 if len(prob_arr) > 3 else 0.0)
            )
            risk_score = round(min(100.0, max(0.0, risk_score)), 1)

            prob_dict = {
                self._mealplan_classes.get(c_idx, str(c_idx)): round(float(p), 4)
                for c_idx, p in enumerate(prob_arr)
            }

            results[orig_idx] = MealPlanRiskPrediction(
                plan_id=item.plan_id,
                facility_id=item.facility_id,
                risk_level=level,
                risk_code=code,
                risk_score=risk_score,
                confidence=round(confidence, 4),
                probabilities=prob_dict,
            )

        return results


@lru_cache(maxsize=1)
def get_cost_risk_service() -> CostRiskModelBundle:
    """애플리케이션 프로세스당 1회 모델 번들을 로드하여 재사용."""
    return CostRiskModelBundle()
