import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging
from Networksecurity.entity.config_entity import DataIngestionConfig
from Networksecurity.entity.artifact_entity import DataIngestionArtifact

# Load environment variables
load_dotenv()
MONGO_DB_URI = os.getenv("MONGO_DB_URI")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """Fetch data from MongoDB collection and return as DataFrame."""
        try:
            database_name = self.data_ingestion_config.data_ingestion_database_name
            collection_name = self.data_ingestion_config.data_ingestion_collection_name

            mongo_client = pymongo.MongoClient(MONGO_DB_URI)
            collection = mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
            df.replace("na", np.nan, inplace=True)

            logging.info(f"Data fetched from MongoDB: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_to_feature_store(self, df: pd.DataFrame) -> str:
        """Save DataFrame to feature store directory as CSV."""
        try:
            feature_store_file_path = os.path.join(
                self.data_ingestion_config.feature_store_dir,
                self.data_ingestion_config.feature_store_file_name
            )
            os.makedirs(self.data_ingestion_config.feature_store_dir, exist_ok=True)
            df.to_csv(feature_store_file_path, index=False, header=True)

            logging.info(f"Data exported to feature store at: {feature_store_file_path}")
            return feature_store_file_path
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame) -> None:
        """Split DataFrame into train and test sets and save them."""
        try:
            train_set, test_set = train_test_split(
                df, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            os.makedirs(self.data_ingestion_config.ingested_dir, exist_ok=True)

            train_file_path = self.data_ingestion_config.train_file_path
            test_file_path = self.data_ingestion_config.test_file_path

            train_set.to_csv(train_file_path, index=False, header=True)
            test_set.to_csv(test_file_path, index=False, header=True)

            logging.info(f"Train/Test split completed. Train: {train_set.shape}, Test: {test_set.shape}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Main pipeline: fetch, store, split, and return artifact."""
        try:
            df = self.export_collection_as_dataframe()
            self.export_data_to_feature_store(df)
            self.split_data_as_train_test(df)

            data_ingestion_artifact = DataIngestionArtifact(
            train_file_path=self.data_ingestion_config.train_file_path,
            test_file_path=self.data_ingestion_config.test_file_path,
            feature_store_file_name=self.data_ingestion_config.feature_store_file_name,
            data_file_path=self.data_ingestion_config.feature_store_dir  # or wherever raw data is stored
)

            logging.info("Data ingestion pipeline completed successfully.")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
