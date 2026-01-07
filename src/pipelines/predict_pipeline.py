import sys
import json
import joblib
import pandas as pd

from src.utils.logger import logging
from src.utils.exception_handler import CustomException


class PredictPipeline:
    def __init__(self, pipeline_path: str, metadata_path: str):
        try:
            logging.info("🚀 Loading inference pipeline")

            self.pipeline = joblib.load(pipeline_path)

            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

            self.threshold = self.metadata.get("threshold", 0.5)

            logging.info(
                f"✅ Pipeline loaded | Threshold={self.threshold}"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_df: pd.DataFrame):
        try:
            logging.info("🔮 Running churn inference")

            probs = self.pipeline.predict_proba(input_df)[:, 1]
            preds = (probs >= self.threshold).astype(int)

            results = pd.DataFrame({
                "churn_probability": probs,
                "churn_prediction": preds
            })

            return results

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    predictor = PredictPipeline(
        pipeline_path="artifacts/churn_pipeline.joblib",
        metadata_path="artifacts/model_metadata.json"
    )

    sample = pd.DataFrame([{
        "Age": 35,
        "Tenure": 12,
        "Usage Frequency": 5,
        "Support Calls": 1,
        "Payment Delay": 0,
        "Total Spend": 450.0,
        "Last Interaction": 10,
        "Gender": "Male",
        "Contract Length": "Monthly",
        "Subscription Type": "Basic"
    }])

    print(predictor.predict(sample))
