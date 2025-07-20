from zenml.client import Client
from mlflow.tracking import MlflowClient, artifact_utils
from steps.celan_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model, experiment_tracker
from zenml import pipeline, step, get_step_context
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowDeploymentConfig

docker_settings = DockerSettings(required_integrations = [MLFLOW])

@step
def deployment_trigger(r2: float, min_accuracy: float)->bool:
    decision = r2 >= min_accuracy
    print("Deploy decision type:", type(decision))
    return bool(decision)

@step(experiment_tracker="mlflow_tracker")
def deploy_model(model, deploy_decision, workers: int, timeout:int)->None:
    if deploy_decision:
        deployer = MLFlowModelDeployer.get_active_model_deployer()

        zenml_client = Client()
        # model_deployer = zenml_client.active_stack_model.model_deployer
        exp_tracker = zenml_client.active_stack.experiment_tracker

        # step_run = get_step_context().step_run.name
        # pipeline_name  = get_step_context().pipeline.name
        #
        # step_run_info = zenml_client.get_pipeline(pipeline_name).get_run().get_step(step_run)
        #
        # mlflow_run_id = exp_tracker.get_step_run_metadata(
        #     step_run_info
        # )

        mlflow_run_id = get_step_context().step_run.id

        # mlflow_run_id = get_step_context().step_run.run_metadata['mlflow.run_id']
        #
        print(f"========================={mlflow_run_id}=============================")

        # exp_tracker.configure_mlflow()
        client = MlflowClient()
        model_name = "model"
        model_uri = artifact_utils.get_artifact_uri(
            run_id = mlflow_run_id,
            artifact_path = model_name
        )

        config = MLFlowDeploymentConfig(
            name = "Trial1",
            # pipeline_name = get_step_context().pipeline,
            model_uri = model_uri,
            model = model,
            model_name = "Linear_regression",
            workers = workers,
            timeout = DEFAULT_SERVICE_START_STOP_TIMEOUT
        )

        deployer.deploy_model(
            config = config,
            service_type = MLFlowDeploymentService.SERVICE_TYPE
        )
        print("Model Deployed successfully")
    else:
        print("Model not deployed due to threshold")

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
        data_path: str,
        min_accuracy: float = 0,
        workers: int=1,
        timeout: int = 300
):
    df = ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, y_train)
    mse, r2 = evaluate_model(model, x_test, y_test)
    deploy_decision = deployment_trigger(r2, min_accuracy)
    deploy_model(model, deploy_decision, workers, timeout)


