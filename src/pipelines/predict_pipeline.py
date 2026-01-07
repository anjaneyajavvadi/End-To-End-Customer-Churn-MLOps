import os
import sys
import joblib
import pandas as pd

from src.utils.logger import logging
from src.utils.exception_handler import CustomException


class PredictPipeline:
    def __init__(
        self,
        model_path: str = "models/model.joblib",
        preprocessor_path: str = "data/features/preprocessor.joblib",
        threshold: float = 0.5,
    ):
        try:
            logging.info("🚀 Loading inference artifacts")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing model → {model_path}")

            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Missing preprocessor → {preprocessor_path}")

            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            self.threshold = threshold

            logging.info("✅ Inference artifacts loaded")

        except Exception as e:
            raise CustomException(e, sys)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        return df

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("🔮 Running churn inference")

            input_df = self._normalize_columns(input_df)

            X = self.preprocessor.transform(input_df)
            probs = self.model.predict_proba(X)[:, 1]
            preds = (probs >= self.threshold).astype(int)

            return pd.DataFrame({
                "churn_probability": probs,
                "churn_prediction": preds,
            })

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    predictor = PredictPipeline()

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
