import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

class LinearRegressionModel(Model):
    def train(self, x_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            logging.info("Model Training Completed")
            return reg
        except Exception as e:
            logging.error(f"Model failed to train: {e}")
            raise  e