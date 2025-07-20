from pipelines.training_pipeline import training_pipeline
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

tracking_uri = get_tracking_uri()

print(f"Tracking_uri: {tracking_uri}")
print(f"Tracker name: {Client().active_stack.experiment_tracker.name}")
training_pipeline(data_path = "C:/Pycharm/MLPipeLineBrazil/data/olist_customers_dataset.csv")
