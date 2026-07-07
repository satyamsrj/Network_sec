from Networksecurity.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import os,sys

from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging


class NetworkModel:
    def __init__(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
        