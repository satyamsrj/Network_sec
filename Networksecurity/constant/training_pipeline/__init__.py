import os
import sys
import numpy as np
import pandas as pd
"""
Data Ingestion related constants are defined in this file. 
This file is created to avoid hardcoding of values.
"""
DATA_INGESTION_COLLECTION_NAME = "phishing"
DATA_INGESTION_DATABASE_NAME = "Networksecurity"
DATA_INGESTION_DIR_NAME = os.path.join("artifacts", "data_ingestion")
DATA_INGESTION_FEATURE_STORE_DIR = os.path.join(DATA_INGESTION_DIR_NAME, "feature_store")
DATA_INGESTION_INGESTED_DIR = os.path.join(DATA_INGESTION_DIR_NAME, "ingested")
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Defining common constants for the project. 
This file is created to avoid hardcoding of values.
"""
TARGET_COLUMN = "status"
PIPELINE_NAME: str = "Networksecurity"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "phishing.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
FEATURE_STORE_FILE_NAME: str = "feature_store.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

# Directory names for data validation artifacts
# Data Validation constants
DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR_NAME = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "drift_report.yaml"   # <-- missing constant
DATA_VALIDATION_VALID_DIR_NAME = "validated"
DATA_VALIDATION_INVALID_DIR_NAME = "invalid"
