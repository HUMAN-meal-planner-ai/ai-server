from fastapi import FastAPI

from app.api.routes.price_prediction import router as price_prediction_router


app = FastAPI(title="MealFit AI Server")
app.include_router(price_prediction_router)
