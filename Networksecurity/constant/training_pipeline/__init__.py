import os
import numpy as np

"""
Data Ingestion related constants
"""
DATA_INGESTION_COLLECTION_NAME = "phishing"
DATA_INGESTION_DATABASE_NAME = "Networksecurity"
DATA_INGESTION_DIR_NAME = os.path.join("artifacts", "data_ingestion")
DATA_INGESTION_FEATURE_STORE_DIR = os.path.join(DATA_INGESTION_DIR_NAME, "feature_store")
DATA_INGESTION_INGESTED_DIR = os.path.join(DATA_INGESTION_DIR_NAME, "ingested")
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Common constants
"""
TARGET_COLUMN = "status"
PIPELINE_NAME: str = "Networksecurity"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "phishing.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
FEATURE_STORE_FILE_NAME: str = "feature_store.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"


"""
Data Validation constants
"""
DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR_NAME = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "drift_report.yaml"
DATA_VALIDATION_VALID_DIR_NAME = "validated"
DATA_VALIDATION_INVALID_DIR_NAME = "invalid"

"""
Data Transformation constants
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Preprocessing object file name
PREPROCESSING_FILE_NAME: str = "preprocessor.pkl"
TARGET_COLUMN = "class"


# KNN imputer parameters
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    'missing_values': np.nan,
    'n_neighbors': 3,
    'weights': 'uniform',
}
