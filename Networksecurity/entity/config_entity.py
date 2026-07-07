from datetime import datetime
import os
from Networksecurity.constant import training_pipeline

print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)

class TrainingPipelineConfig:
    def __init__(self,time_stamp: str = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")):
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name,time_stamp)
        self.time_stamp = time_stamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.data_ingestion_database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME)

        # Feature store directory + file name
        self.feature_store_dir = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR)
        self.feature_store_file_name = training_pipeline.FEATURE_STORE_FILE_NAME   # <-- ADD THIS

        # Train/Test split directories
        self.ingested_dir = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR)
        self.train_file_path = os.path.join(self.ingested_dir, training_pipeline.TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(self.ingested_dir, training_pipeline.TEST_FILE_NAME)

        self.train_test_split_ratio = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

class DataValidationConfig:


    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
        self.valid_data_dir = os.path.join(self.data_validation_dir, "valid_data")
        self.invalid_data_dir = os.path.join(self.data_validation_dir, "invalid_data")
        self.valid_train_file_path = os.path.join(self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path = os.path.join(self.valid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path = os.path.join(self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path = os.path.join(self.invalid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.drift_report_file_path = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR_NAME,
                                   training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
        
    
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Base directory for data transformation artifacts
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        # Directory for transformed data
        transformed_data_dir = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
        )

        # Directory for transformed objects (preprocessor, etc.)
        transformed_object_dir = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR
        )

        # Paths for train/test numpy arrays
        self.transformed_train_file_path: str = os.path.join(
            transformed_data_dir,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy")
        )

        self.transformed_test_file_path: str = os.path.join(
            transformed_data_dir,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy")
        )

        # Path for the preprocessing object (pickle/dill)
        self.transformed_object_file_path: str = os.path.join(
            transformed_object_dir,
            training_pipeline.PREPROCESSING_FILE_NAME
        )


class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir:str = os.path.join(
            training_pipeline_config.artifact_dir,training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path:str = os.path.join(
            self.model_trainer_dir,training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,training_pipeline.MODEL_FILE_NAME
        )
        self.expected_accuracy:float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
        