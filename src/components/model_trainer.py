import os
import sys
import json
from dataclasses import dataclass

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

import mlflow
import mlflow.sklearn

from src.utils.logger import logging
from src.utils.exception_handler import CustomException


@dataclass
class ModelTrainerConfig:
    features_dir: str = os.path.join("data", "features")
    models_dir: str = "models"
    metrics_path: str = "metrics.json"

    X_train_path: str = os.path.join("data", "features", "X_train.npy")
    y_train_path: str = os.path.join("data", "features", "y_train.npy")
    X_test_path: str = os.path.join("data", "features", "X_test.npy")
    y_test_path: str = os.path.join("data", "features", "y_test.npy")

    model_path: str = os.path.join("models", "model.joblib")

    random_state: int = 42
    max_iter: int = 1000


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def _load_array(self, path: str, name: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} file → {path}")
        return np.load(path)

    def run(self):
        try:
            logging.info("🚀 Starting model training stage")

        
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("churn_prediction")

            with mlflow.start_run():

            
                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_param("max_iter", self.config.max_iter)
                mlflow.log_param("class_weight", "balanced")
                mlflow.log_param("random_state", self.config.random_state)

            
                X_train = self._load_array(self.config.X_train_path, "X_train")
                y_train = self._load_array(self.config.y_train_path, "y_train")
                X_test = self._load_array(self.config.X_test_path, "X_test")
                y_test = self._load_array(self.config.y_test_path, "y_test")

                logging.info(
                    f"Loaded shapes → "
                    f"X_train={X_train.shape}, "
                    f"X_test={X_test.shape}"
                )

            
                model = LogisticRegression(
                    max_iter=self.config.max_iter,
                    class_weight="balanced",
                    random_state=self.config.random_state,
                )

                model.fit(X_train, y_train)

            
                probs = model.predict_proba(X_test)[:, 1]

                roc_auc = roc_auc_score(y_test, probs)
                pr_auc = average_precision_score(y_test, probs)

                logging.info(f"ROC-AUC = {roc_auc:.4f}")
                logging.info(f"PR-AUC  = {pr_auc:.4f}")

            
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_metric("pr_auc", pr_auc)

            
                os.makedirs(self.config.models_dir, exist_ok=True)

                joblib.dump(model, self.config.model_path)

                metrics = {
                    "roc_auc": float(roc_auc),
                    "pr_auc": float(pr_auc),
                }

                with open(self.config.metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)

                # log artifacts to MLflow
                mlflow.log_artifact(self.config.metrics_path)
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name="ChurnModel"
                )

                logging.info("✅ Model training completed successfully")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ModelTrainer().run()
