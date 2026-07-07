import os,sys


from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging
from Networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from Networksecurity.entity.config_entity import ModelTrainerConfig

from Networksecurity.utils.ml_utils.model.estimator import NetworkMOdel
from Networksecurity.utils.ml_utils.metric.classification import get_classification_score
from Networksecurity.utils.main_utils.utils import save_object,load_object
from Networksecurity.utils.main_utils.utils import load_numpy_array_data

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e :
            raise NetworkSecurityException(e,sys)
        
    def intiate_model_trainer(self) -> ModelTrainerArtifact:
        
        