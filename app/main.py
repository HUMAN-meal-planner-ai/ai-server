from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.menu_search import router as menu_search_router
from app.api.routes.price_prediction import router as price_prediction_router

app = FastAPI(title="MealFit AI Server")

# Frontend 개발 서버에서 AI Server API를 호출할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# 메뉴 검색 API
app.include_router(menu_search_router)

# 가격 예측 API
app.include_router(price_prediction_router)
