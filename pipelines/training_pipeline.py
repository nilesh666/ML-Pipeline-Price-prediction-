from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.celan_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline
def training_pipeline(data_path: str):
    df = ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, y_train)
    mse, r2_score = evaluate_model(model, x_test, y_test)

