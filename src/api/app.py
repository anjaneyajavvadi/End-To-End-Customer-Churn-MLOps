from fastapi import FastAPI, HTTPException

from src.api.schema import ChurnRequest, ChurnResponse
from src.api.predictor import ChurnPredictor
from src.utils.logger import logging


app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
)

predictor = ChurnPredictor()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    try:
        prob, pred = predictor.predict(request.dict())
        return {
            "churn_probability": prob,
            "churn_prediction": pred,
        }
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")
