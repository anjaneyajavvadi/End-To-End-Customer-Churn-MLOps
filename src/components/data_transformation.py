import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils.exception_handler import CustomException
from src.utils.logger import logging


@dataclass
class DataTransformationConfig:
    target_column: str = "churn"


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self):
        try:
            numeric_features = [
                    "age",
                    "tenure",
                    "usage_frequency",
                    "support_calls",
                    "payment_delay",
                    "total_spend",
                    "last_interaction",
                ]

            categorical_features = [
                    "gender",
                    "contract_length",
                    "subscription_type",
                ]
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_features),
                    ("cat", categorical_pipeline, categorical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            logging.info("🚀 Starting data transformation")

            X_train = train_df.drop(columns=[self.config.target_column])
            y_train = train_df[self.config.target_column]

            X_test = test_df.drop(columns=[self.config.target_column])
            y_test = test_df[self.config.target_column]

            preprocessor = self.get_preprocessor()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("✅ Data transformation completed")

            return (
                X_train_transformed,
                y_train.values,
                X_test_transformed,
                y_test.values,
                preprocessor
            )

        except Exception as e:
            raise CustomException(e, sys)
