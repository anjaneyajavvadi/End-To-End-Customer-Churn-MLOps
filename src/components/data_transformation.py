import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

from src.utils.logger import logging
from src.utils.exception_handler import CustomException


@dataclass
class DataTransformationConfig:
    processed_dir: str = os.path.join("data", "processed")
    features_dir: str = os.path.join("data", "features")

    train_input_path: str = os.path.join("data", "processed", "train.csv")
    test_input_path: str = os.path.join("data", "processed", "test.csv")

    X_train_path: str = os.path.join("data", "features", "X_train.npy")
    y_train_path: str = os.path.join("data", "features", "y_train.npy")
    X_test_path: str = os.path.join("data", "features", "X_test.npy")
    y_test_path: str = os.path.join("data", "features", "y_test.npy")

    preprocessor_path: str = os.path.join("data", "features", "preprocessor.joblib")

    target_column: str = "churn"


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

        self.numeric_features = [
            "age",
            "tenure",
            "usage_frequency",
            "support_calls",
            "payment_delay",
            "total_spend",
            "last_interaction",
        ]

        self.categorical_features = [
            "gender",
            "contract_length",
            "subscription_type",
        ]

    def _build_preprocessor(self):
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ]
        )

    def _load_data(self, path: str, split_name: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split_name}: Missing file → {path}")

        logging.info(f"{split_name}: Loading {path}")
        return pd.read_csv(path)

    def run(self):
        try:
            logging.info("🚀 Starting data transformation stage")

            train_df = self._load_data(self.config.train_input_path, "TRAIN")
            test_df = self._load_data(self.config.test_input_path, "TEST")

            X_train = train_df.drop(columns=[self.config.target_column])
            y_train = train_df[self.config.target_column].values

            X_test = test_df.drop(columns=[self.config.target_column])
            y_test = test_df[self.config.target_column].values

            preprocessor = self._build_preprocessor()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info(
                f"Transformed shapes → "
                f"X_train={X_train_transformed.shape}, "
                f"X_test={X_test_transformed.shape}"
            )

            os.makedirs(self.config.features_dir, exist_ok=True)

            np.save(self.config.X_train_path, X_train_transformed)
            np.save(self.config.y_train_path, y_train)
            np.save(self.config.X_test_path, X_test_transformed)
            np.save(self.config.y_test_path, y_test)

            joblib.dump(preprocessor, self.config.preprocessor_path)

            logging.info("✅ Data transformation completed successfully")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataTransformation().run()
