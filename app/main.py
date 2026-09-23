from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.menu_search import router as menu_search_router

app = FastAPI(title="MealFit AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

app.include_router(menu_search_router)

