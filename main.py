from Networksecurity.components.data_ingestion import DataIngestion
from Networksecurity.components.data_validation import DataValidation
from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, TrainingPipelineConfig
from Networksecurity.logging.logger import logging
import sys

if __name__ == "__main__":
    try:
        # Initialize pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)

        # Run data ingestion
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Starting data ingestion process...")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()


        # Log artifact details
        logging.info(f"Data ingestion completed successfully.")

        # Initialize data validation
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
        logging.info("Starting data validation process...")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)



    except Exception as e:
        raise NetworkSecurityException(e, sys)
