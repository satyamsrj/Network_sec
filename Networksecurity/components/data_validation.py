import os, sys
import pandas as pd
from scipy.stats import ks_2samp

from Networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from Networksecurity.entity.config_entity import DataValidationConfig
from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging
from Networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file

# Make sure SCHEMA_FILE_PATH is imported from your constants/config
from Networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("DataValidation initialized successfully.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Reads CSV file into a DataFrame."""
        try:
            logging.info(f"Reading data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """Checks if dataframe has the required number of columns."""
        try:
            required_columns = list(self.schema_config["columns"].keys())
            actual_columns = list(dataframe.columns)
            missing = set(required_columns) - set(actual_columns)
            extra = set(actual_columns) - set(required_columns)

            if missing:
                logging.error(f"Missing columns: {missing}")
            if extra:
                logging.error(f"Unexpected columns: {extra}")

            return not missing and not extra
        except Exception as e:
                raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """Detects drift between base and current dataset using KS test."""
        try:
            report = {}
            drift_overall_status = True

            for column in base_df.columns:
                d1, d2 = base_df[column], current_df[column]
                p_value = ks_2samp(d1, d2).pvalue
                drift_status = p_value >= threshold

                if not drift_status:
                    drift_overall_status = False

                report[column] = {"p_value": float(p_value), "drift_status": drift_status}

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            logging.info(f"Drift detection completed. Overall status: {drift_overall_status}")
            return drift_overall_status
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Runs validation checks and returns DataValidationArtifact."""
        try:
            train_df = DataValidation.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = DataValidation.read_data(self.data_ingestion_artifact.test_file_path)

            # Validate schema
            if not self.validate_number_of_columns(train_df):
                raise Exception("Train dataset does not contain all required columns.")
            if not self.validate_number_of_columns(test_df):
                raise Exception("Test dataset does not contain all required columns.")

            # Detect drift
            drift_status = self.detect_dataset_drift(train_df, test_df)

            # Save validated datasets
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact created: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
