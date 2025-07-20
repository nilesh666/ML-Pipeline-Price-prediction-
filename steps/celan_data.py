import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy

@step
def clean_data(df: pd.DataFrame)-> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()

        logging.info("Data cleaning completed!!!")

        return x_train, x_test, y_train, y_test


    except Exception as e:
        logging.error(f"Error in clean_data: {e}")
        raise e

