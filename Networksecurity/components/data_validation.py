from Networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from Networksecurity.entity.config_entity import DataValidationConfig
from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging

from scipy.stats import ks_2samp
import os,sys
import pandas as pd
from Networksecurity.utils.main_utils.utils import read_yaml_file , write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self.schema_config)
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_datset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        try:
            report = {}
            status = True
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                p_value = ks_2samp(d1, d2).pvalue
                if p_value < threshold:
                    status = False
                else:
                    status = True
                report.update({column: {"p_value": float(p_value), "drift_status": status}})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)   
            write_yaml_file(file_path=drift_report_file_path, content=report)     
            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            training_file_path = self.data_ingestion_artifact.trained_file_path
            testing_file_path = self.data_ingestion_artifact.test_file_path
            train_data_frame = DataValidation.read_data(training_file_path)
            test_data_frame = DataValidation.read_data(testing_file_path)
            status = self.validate_number_of_columns(dataframe=train_data_frame)
            if not status:
                raise Exception(f"Train dataset does not contain all columns.")
            status = self.validate_number_of_columns(dataframe=test_data_frame) 
            if not status:
                raise Exception(f"Test dataset does not contain all columns.")
            # Check data drift
            drift_status = self.detect_datset_drift(base_df=train_data_frame, current_df=test_data_frame)
            if not drift_status:
                raise Exception(f"Data drift detected between train and test datasets.")
            
            ## Check Data Drift
            status = self.detect_datset_drift(base_df=train_data_frame, current_df=test_data_frame)
            dir_path = os.path.dirname(self.data_validation_config.drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )



        except Exception as e:
            raise NetworkSecurityException(e, sys)