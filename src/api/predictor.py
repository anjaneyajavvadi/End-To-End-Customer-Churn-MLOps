import os
import joblib
import pandas as pd

from src.utils.logger import logging


class ChurnPredictor:
    def __init__(
        self,
        model_path="models/model.joblib",
        preprocessor_path="data/features/preprocessor.joblib",
        threshold=0.5,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model → {model_path}")

        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Missing preprocessor → {preprocessor_path}")

        logging.info("🚀 Loading model & preprocessor")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.threshold = threshold

        logging.info("✅ Model & preprocessor loaded")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        return df

    def predict(self, data: dict):
        df = pd.DataFrame([data])
        df = self._normalize_columns(df)

        X = self.preprocessor.transform(df)
        prob = float(self.model.predict_proba(X)[0, 1])
        pred = int(prob >= self.threshold)

        return prob, pred
