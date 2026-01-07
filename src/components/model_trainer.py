import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils.logger import logging
from src.utils.exception_handler import CustomException


@dataclass
class ModelTrainerConfig:
    random_state: int = 42


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        try:
            logging.info("🚀 Training churn model")

            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.config.random_state
            )

            model.fit(X_train, y_train)

            probs = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)

            logging.info(f"ROC-AUC: {roc_auc:.4f}")
            logging.info(f"PR-AUC: {pr_auc:.4f}")

            metrics = {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc
            }

            return model, metrics

        except Exception as e:
            raise CustomException(e, sys)
