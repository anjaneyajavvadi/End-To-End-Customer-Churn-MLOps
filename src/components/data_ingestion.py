import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.utils.logger import logging
from src.utils.exception_handler import CustomException


@dataclass
class DataIngestionConfig:
    raw_train_path: str = os.path.join("data", "raw", "train.csv")
    raw_test_path: str = os.path.join("data", "raw", "test.csv")
    processed_dir: str = os.path.join("data", "processed")
    processed_train_path: str = os.path.join("data", "processed", "train.csv")
    processed_test_path: str = os.path.join("data", "processed", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def _validate_raw_data(self, df: pd.DataFrame):
        """Basic schema validation."""
        required_columns = {
            "Age",
            "Tenure",
            "Usage Frequency",
            "Support Calls",
            "Payment Delay",
            "Total Spend",
            "Last Interaction",
            "Gender",
            "Contract Length",
            "Subscription Type",
            "Churn"
        }

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    def _handle_missing_target(self, df: pd.DataFrame, split_name: str):
        missing = df["Churn"].isna().sum()
        if missing > 0:
            logging.warning(
                f"{split_name}: Dropping {missing} rows with missing Churn labels"
            )
            df = df.dropna(subset=["Churn"])
        return df


    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        return df

    def run(self):
        try:
            logging.info("🚀 Starting data ingestion (pre-split mode)")

            if not os.path.exists(self.config.raw_train_path):
                raise FileNotFoundError(f"Missing {self.config.raw_train_path}")

            if not os.path.exists(self.config.raw_test_path):
                raise FileNotFoundError(f"Missing {self.config.raw_test_path}")

            train_df = pd.read_csv(self.config.raw_train_path)
            test_df = pd.read_csv(self.config.raw_test_path)

            train_df = self._handle_missing_target(train_df, "TRAIN")
            test_df = self._handle_missing_target(test_df, "TEST")

            train_df = self._normalize_columns(train_df)
            test_df = self._normalize_columns(test_df)

            logging.info(
                f"Loaded train shape={train_df.shape}, test shape={test_df.shape}"
            )

            os.makedirs(self.config.processed_dir, exist_ok=True)
            train_df.to_csv(self.config.processed_train_path, index=False)
            test_df.to_csv(self.config.processed_test_path, index=False)

            return (
                self.config.processed_train_path,
                self.config.processed_test_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.run()
    print("✅ Data ingestion completed")
    print("Train:", train_path)
    print("Test:", test_path)
