import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def train_model(
        x_train: pd.DataFrame,
        #x_test: pd.DataFrame,
        y_train: pd.Series,
        #y_test: pd.Series
)->RegressorMixin:
    try:
        model = LinearRegressionModel()
        trained_model = model.train(x_train, y_train)
        mlflow.sklearn.log_model(model, artifact_path="model")
        return trained_model
    except Exception as e:
        logging.error(f"Falied to train the model model_train.py: {e}")
        raise e

