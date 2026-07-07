import sys
import os
from Networksecurity.components.data_ingestion import DataIngestion
from Networksecurity.components.data_validation import DataValidation
from Networksecurity.components.data_transformation import DataTransformation
from Networksecurity.components.model_trainer import ModelTrainer
from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from Networksecurity.logging.logger import logging


def get_latest_run_artifact(base_dir="artifacts"):
    """Helper to get the latest timestamped run folder."""
    runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not runs:
        raise Exception("No run folders found in artifacts directory.")
    latest_run = sorted(runs)[-1]  # pick most recent timestamp
    return os.path.join(base_dir, latest_run)


if __name__ == "__main__":
    try:
        # Step 1: Initialize pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()

        # Step 2: Data Ingestion
        logging.info("Starting data ingestion process...")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully.")
        print(data_ingestion_artifact)

        # Step 3: Data Validation
        logging.info("Starting data validation process...")
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed successfully.")
        print(data_validation_artifact)

        # Step 4: Data Transformation
        logging.info("Starting data transformation process...")
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed successfully.")
        print(data_transformation_artifact)

        # Step 5: Model Training
        logging.info("Starting model training process...")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed successfully.")
        print(model_trainer_artifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
