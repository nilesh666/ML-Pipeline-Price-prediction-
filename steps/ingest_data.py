import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path:str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    try:
        ingest = IngestData(data_path)
        df = ingest.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e


