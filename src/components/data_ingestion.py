import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.utils.logger import logging
from src.utils.exception_handler import CustomException


@dataclass
class DataIngestionConfig:
    raw_dir: str = os.path.join("data", "raw")
    processed_dir: str = os.path.join("data", "processed")

    raw_train_path: str = os.path.join("data", "raw", "train.csv")
    raw_test_path: str = os.path.join("data", "raw", "test.csv")

    processed_train_path: str = os.path.join("data", "processed", "train.csv")
    processed_test_path: str = os.path.join("data", "processed", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

        # schema AFTER normalization
        self.required_columns = {
            "age",
            "tenure",
            "usage_frequency",
            "support_calls",
            "payment_delay",
            "total_spend",
            "last_interaction",
            "gender",
            "contract_length",
            "subscription_type",
            "churn",
        }

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        return df

    def _validate_schema(self, df: pd.DataFrame, split_name: str):
        missing = self.required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"{split_name}: Missing required columns: {missing}"
            )

    def _handle_missing_target(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=["churn"])
        after = len(df)

        if before != after:
            logging.warning(
                f"{split_name}: Dropped {before - after} rows with missing churn"
            )

        return df

    def _load_csv(self, path: str, split_name: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split_name}: File not found → {path}")

        logging.info(f"{split_name}: Loading data from {path}")
        return pd.read_csv(path)

    def run(self):
        try:
            logging.info("🚀 Starting data ingestion stage")

            # load
            train_df = self._load_csv(self.config.raw_train_path, "TRAIN")
            test_df = self._load_csv(self.config.raw_test_path, "TEST")

            # normalize
            train_df = self._normalize_columns(train_df)
            test_df = self._normalize_columns(test_df)

            # validate
            self._validate_schema(train_df, "TRAIN")
            self._validate_schema(test_df, "TEST")

            # clean
            train_df = self._handle_missing_target(train_df, "TRAIN")
            test_df = self._handle_missing_target(test_df, "TEST")

            logging.info(
                f"TRAIN shape={train_df.shape} | TEST shape={test_df.shape}"
            )

            # write
            os.makedirs(self.config.processed_dir, exist_ok=True)

            train_df.to_csv(self.config.processed_train_path, index=False)
            test_df.to_csv(self.config.processed_test_path, index=False)

            logging.info("✅ Data ingestion completed successfully")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataIngestion().run()
