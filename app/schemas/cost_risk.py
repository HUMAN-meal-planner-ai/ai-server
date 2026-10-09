from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# 메뉴 가격 위험도 분류 (Menu Risk Classification)
# ------------------------------------------------------------------

class MenuRiskFeatures(BaseModel):
    """메뉴 위험도 분류 모델 입력 피처."""
    menu_id: Optional[int] = Field(None, description="메뉴 ID")
    category_code: int = Field(0, description="대분류 인코딩 코드")
    ingredient_count: int = Field(0, ge=0, description="메뉴 구성 식재료 수")
    cur_menu_cost: float = Field(0.0, ge=0, description="현재 1인분 메뉴 원가 (원)")
    menu_cost_increase_rate: float = Field(0.0, description="7일 후 메뉴 총 원가 변동률 (%)")
    max_ingredient_increase_rate: float = Field(0.0, description="최고 상승 식재료 단가 상승률 (%)")
    max_contribution_rate: float = Field(0.0, ge=0, le=100, description="최고 상승 식재료의 원가 기여도 (%)")
    top_driver_risk: float = Field(0.0, description="Top Driver 위험 지수 (상승률 * 기여도 / 100)")
    high_risk_ingredient_count: int = Field(0, ge=0, description="15% 이상 급등 식재료 수")
    caution_ingredient_count: int = Field(0, ge=0, description="5%~15% 상승 식재료 수")
    avg_price_volatility: float = Field(0.05, ge=0, description="식재료 30일 평균 가격 변동성")


class MenuRiskPrediction(BaseModel):
    """메뉴 위험도 분류 결과."""
    menu_id: Optional[int] = None
    risk_level: str = Field(..., description="위험 등급: SAFE, CAUTION, WARNING")
    risk_code: int = Field(..., description="위험 등급 코드 (0: SAFE, 1: CAUTION, 2: WARNING)")
    risk_score: float = Field(..., ge=0, le=100, description="100점 만점 환산 위험 점수")
    confidence: float = Field(..., ge=0, le=1.0, description="모델 예측 신뢰도")
    probabilities: dict[str, float] = Field(..., description="각 등급별 예측 확률")


class MenuRiskBatchRequest(BaseModel):
    """메뉴 위험도 배치 예측 요청."""
    items: List[MenuRiskFeatures]


class MenuRiskBatchResponse(BaseModel):
    """메뉴 위험도 배치 예측 응답."""
    predictions: List[MenuRiskPrediction]


# ------------------------------------------------------------------
# 식단 종합 위험도 분류 (MealPlan Risk Classification)
# ------------------------------------------------------------------

class MealPlanRiskFeatures(BaseModel):
    """식단 종합 위험도 분류 모델 입력 피처."""
    plan_id: Optional[int] = Field(None, description="식단 ID")
    facility_id: Optional[int] = Field(None, description="시설 ID")
    total_menu_count: int = Field(0, ge=0, description="식단 내 총 메뉴 수") # 최솟값 완화, 결측치 허용
    warning_menu_ratio: float = Field(0.0, ge=0, le=100, description="WARNING 메뉴 비율 (%)")
    caution_menu_ratio: float = Field(0.0, ge=0, le=100, description="CAUTION 메뉴 비율 (%)")
    risk_cost_ratio: float = Field(0.0, ge=0, le=100, description="위험 메뉴 비용 비중 (%)")
    warning_cost_ratio: float = Field(0.0, ge=0, le=100, description="경고 메뉴 비용 비중 (%)")
    total_expected_cost: float = Field(0.0, ge=0, description="식단 총 예상 소요 원가 (원)")
    budget_usage_ratio: float = Field(0.0, ge=0, description="월 예산 대비 식단 소요 비율 (%)")
    avg_menu_cost_increase_rate: float = Field(0.0, description="메뉴 평균 원가 변동률 (%)")
    max_menu_cost_increase_rate: float = Field(0.0, description="메뉴 최고 원가 변동률 (%)")


class MealPlanRiskPrediction(BaseModel):
    """식단 종합 위험도 분류 결과."""
    plan_id: Optional[int] = None
    facility_id: Optional[int] = None
    risk_level: str = Field(..., description="종합 위험 등급: SAFE, CAUTION, WARNING, CRITICAL")
    risk_code: int = Field(..., description="위험 등급 코드 (0: SAFE, 1: CAUTION, 2: WARNING, 3: CRITICAL)")
    risk_score: float = Field(..., ge=0, le=100, description="100점 만점 환산 종합 위험 점수")
    confidence: float = Field(..., ge=0, le=1.0, description="모델 예측 신뢰도")
    probabilities: dict[str, float] = Field(..., description="각 등급별 예측 확률")


class MealPlanRiskBatchRequest(BaseModel):
    """식단 종합 위험도 배치 예측 요청."""
    items: List[MealPlanRiskFeatures]


class MealPlanRiskBatchResponse(BaseModel):
    """식단 종합 위험도 배치 예측 응답."""
    predictions: List[MealPlanRiskPrediction]
