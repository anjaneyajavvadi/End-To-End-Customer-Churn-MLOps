import os
import sys
import json
import joblib
from datetime import datetime, timezone

from sklearn.pipeline import Pipeline

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.logger import logging
from src.utils.exception_handler import CustomException


class TrainerPipeline:
    def __init__(self):
        self.artifacts_dir = "artifacts"
        self.model_path = os.path.join(self.artifacts_dir, "churn_pipeline.joblib")
        self.metadata_path = os.path.join(self.artifacts_dir, "model_metadata.json")

        os.makedirs(self.artifacts_dir, exist_ok=True)

    def run(self):
        try:
            logging.info("🚀 Starting training pipeline")

            ingestion = DataIngestion()
            train_path, test_path = ingestion.run()

            import pandas as pd
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            transformer = DataTransformation()
            (
                X_train,
                y_train,
                X_test,
                y_test,
                preprocessor,
            ) = transformer.run(train_df, test_df)

            trainer = ModelTrainer()
            model, metrics = trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            joblib.dump(pipeline, self.model_path)
            logging.info(f"✅ Model pipeline saved at: {self.model_path}")

            metadata = {
                "model_type": model.__class__.__name__,
                "metrics": metrics,
                "threshold": 0.5,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }

            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logging.info(f"✅ Metadata saved at: {self.metadata_path}")

            return {
                "model_path": self.model_path,
                "metadata_path": self.metadata_path,
                "metrics": metrics,
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainerPipeline()
    artifacts = pipeline.run()
    print("🎯 Training completed successfully")
    print(artifacts)
