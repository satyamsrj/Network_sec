from Networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
import os, sys
from Networksecurity.exception.exception import NetworkSecurityException

class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            # Store the preprocessor and model for later use
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def predict(self, x):
        try:
            # Transform input using preprocessor
            x_transform = self.preprocessor.transform(x)
            # Predict using trained model
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)
