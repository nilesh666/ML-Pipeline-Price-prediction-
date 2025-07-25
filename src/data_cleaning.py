import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, df: pd.DataFrame)->Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, df: pd.DataFrame) ->pd.DataFrame:
        try:
            data = df.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp"
            ], axis=1)

            # data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace = True)
            # data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace = True)
            # data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace = True)
            # data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace = True)
            # data["review_comment_message"].fillna("No review", inplace = True)
            data["product_weight_g"] = data["product_weight_g"].fillna(data["product_weight_g"].median())
            data["product_length_cm"] = data["product_length_cm"].fillna(data["product_length_cm"].median())
            data["product_height_cm"] = data["product_height_cm"].fillna(data["product_height_cm"].median())
            data["product_width_cm"] = data["product_width_cm"].fillna(data["product_width_cm"].median())
            data["review_comment_message"] = data["review_comment_message"].fillna("No review")

            data = data.select_dtypes(include = [np.number])

            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis = 1)
            return data

        except Exception as e:
            logging.error(f"Error preprocessing the data: {e}")
            raise e

class DataDivideStrategy(DataStrategy):
    def handle_data(self, df: pd.DataFrame):
        try:
            x = df.drop(["review_score"], axis = 1)
            y = df["review_score"]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
            return x_train, x_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error in dividing the data: {e}")
            raise e

class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self):
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
