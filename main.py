from Networksecurity.components.data_ingestion import DataIngestion
from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
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
        logging.info(f"Train file path: {data_ingestion_artifact.train_file_path}")
        logging.info(f"Test file path: {data_ingestion_artifact.test_file_path}")

    except Exception as e:
        raise NetworkSecurityException(e, sys)
