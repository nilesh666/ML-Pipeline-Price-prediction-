import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(
        model: RegressorMixin,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame
)-> Tuple[
    Annotated[float, "mse"],
    Annotated[float, "r2_score"]
]:
    try:
        predictions = model.predict(x_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("mse", mse)
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("r2_score", r2)

        return mse, r2

        # with mlflow.start_run():
        #     predictions = model.predict(x_test)
        #     mse_class = MSE()
        #     mse = mse_class.calculate_scores(y_test, predictions)
        #     mlflow.log_metric("mse", mse)
        #
        #     r2_class = R2()
        #     r2 = r2_class.calculate_scores(y_test, predictions)
        #     mlflow.log_metric("r2_score", r2)
        #     return mse, r2

    except Exception as e:
        logging.error(f"Error in steps\evaluation {e}")
        raise e