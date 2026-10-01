from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.prediction.cost_risk_service import (
    CostRiskModelBundle,
    get_cost_risk_service,
)
from app.schemas.cost_risk import (
    MenuRiskBatchRequest,
    MenuRiskBatchResponse,
    MenuRiskFeatures,
    MenuRiskPrediction,
    MealPlanRiskBatchRequest,
    MealPlanRiskBatchResponse,
    MealPlanRiskFeatures,
    MealPlanRiskPrediction,
)

router = APIRouter(
    prefix="/api/v1/cost-risk",
    tags=["cost-risk"],
)


@router.get("/metadata")
def get_model_metadata(
    service: CostRiskModelBundle = Depends(get_cost_risk_service),
) -> dict:
    """원가 및 식단 종합 위험도 분류 모델 메타데이터를 반환합니다."""
    return service.metadata


# ------------------------------------------------------------------
# 메뉴 가격 위험도 예측 API
# ------------------------------------------------------------------

@router.post(
    "/menus/predict",
    response_model=MenuRiskPrediction,
    summary="단일 메뉴 가격 위험도 분류 예측",
)
def predict_single_menu_risk(
    features: MenuRiskFeatures,
    service: CostRiskModelBundle = Depends(get_cost_risk_service),
) -> MenuRiskPrediction:
    """단일 메뉴의 피처 데이터를 입력받아 위험도(SAFE, CAUTION, WARNING) 및 점수를 예측합니다."""
    results = service.predict_menu_risks([features])
    if not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="메뉴 위험도 예측에 실패했습니다.",
        )
    return results[0]


@router.post(
    "/menus/batch-predict",
    response_model=MenuRiskBatchResponse,
    summary="다중 메뉴 가격 위험도 일괄 예측",
)
def predict_batch_menu_risks(
    request: MenuRiskBatchRequest,
    service: CostRiskModelBundle = Depends(get_cost_risk_service),
) -> MenuRiskBatchResponse:
    """여러 메뉴의 피처 목록을 입력받아 위험도 분류 결과를 일괄 반환합니다."""
    results = service.predict_menu_risks(request.items)
    return MenuRiskBatchResponse(predictions=results)


# ------------------------------------------------------------------
# 식단표 종합 위험도 예측 API
# ------------------------------------------------------------------

@router.post(
    "/meal-plans/predict",
    response_model=MealPlanRiskPrediction,
    summary="단일 식단표 종합 위험도 분류 예측",
)
def predict_single_mealplan_risk(
    features: MealPlanRiskFeatures,
    service: CostRiskModelBundle = Depends(get_cost_risk_service),
) -> MealPlanRiskPrediction:
    """식단표의 종합 피처 데이터를 입력받아 종합 위험도(SAFE, CAUTION, WARNING, CRITICAL) 및 점수를 예측합니다."""
    results = service.predict_mealplan_risks([features])
    if not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="식단 종합 위험도 예측에 실패했습니다.",
        )
    return results[0]


@router.post(
    "/meal-plans/batch-predict",
    response_model=MealPlanRiskBatchResponse,
    summary="다중 식단표 종합 위험도 일괄 예측",
)
def predict_batch_mealplan_risks(
    request: MealPlanRiskBatchRequest,
    service: CostRiskModelBundle = Depends(get_cost_risk_service),
) -> MealPlanRiskBatchResponse:
    """여러 식단표의 종합 피처 목록을 입력받아 일괄 위험도 분류 결과를 반환합니다."""
    results = service.predict_mealplan_risks(request.items)
    return MealPlanRiskBatchResponse(predictions=results)
