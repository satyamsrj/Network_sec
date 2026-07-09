import os
import sys
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from Networksecurity.constant.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
)

from Networksecurity.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)

from Networksecurity.entity.config_entity import (
    DataTransformationConfig,
)

from Networksecurity.exception.exception import (
    NetworkSecurityException,
)

from Networksecurity.logging.logger import logging

from Networksecurity.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object,
)


class DataTransformation:

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):

        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:

        return pd.read_csv(file_path)

    def get_data_transformer_object(self) -> Pipeline:

        logging.info("Initializing KNN Imputer Pipeline")

        preprocessor = Pipeline(
            steps=[
                (
                    "imputer",
                    KNNImputer(
                        **DATA_TRANSFORMATION_IMPUTER_PARAMS
                    ),
                )
            ]
        )

        return preprocessor

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:

        try:

            logging.info("Starting Data Transformation")

            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )

            test_df = self.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            ####################################################
            # Drop unwanted columns
            ####################################################

            for df in [train_df, test_df]:

                drop_cols = []

                for col in ["Index", "Unnamed: 0"]:

                    if col in df.columns:
                        drop_cols.append(col)

                if drop_cols:
                    logging.info(f"Dropping columns : {drop_cols}")
                    df.drop(columns=drop_cols, inplace=True)

            logging.info(f"Train Shape : {train_df.shape}")
            logging.info(f"Test Shape : {test_df.shape}")

            ####################################################
            # Split Features & Target
            ####################################################

            input_feature_train_df = train_df.drop(
                columns=[TARGET_COLUMN],
                axis=1,
            )

            target_feature_train_df = (
                train_df[TARGET_COLUMN]
                .replace(-1, 0)
            )

            input_feature_test_df = test_df.drop(
                columns=[TARGET_COLUMN],
                axis=1,
            )

            target_feature_test_df = (
                test_df[TARGET_COLUMN]
                .replace(-1, 0)
            )

            logging.info(
                f"Training Features Shape : {input_feature_train_df.shape}"
            )

            logging.info(
                f"Testing Features Shape : {input_feature_test_df.shape}"
            )

            logging.info(
                f"Training Columns : {list(input_feature_train_df.columns)}"
            )

            ####################################################
            # Create Preprocessor
            ####################################################

            preprocessor = self.get_data_transformer_object()

            ####################################################
            # Fit & Transform
            ####################################################

            logging.info("Fitting Preprocessor...")

            transformed_train = preprocessor.fit_transform(
                input_feature_train_df
            )

            logging.info("Transforming Test Data...")

            transformed_test = preprocessor.transform(
                input_feature_test_df
            )

            ####################################################
            # Combine X & y
            ####################################################

            train_arr = np.c_[
                transformed_train,
                np.array(target_feature_train_df),
            ]

            test_arr = np.c_[
                transformed_test,
                np.array(target_feature_test_df),
            ]

            ####################################################
            # Save Arrays
            ####################################################

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr,
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr,
            )

            ####################################################
            # Save Preprocessor
            ####################################################

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )

            os.makedirs(
                "final_models",
                exist_ok=True,
            )

            save_object(
                "final_models/preprocessor.pkl",
                preprocessor,
            )

            logging.info("Preprocessor saved successfully.")

            ####################################################
            # Create Artifact
            ####################################################

            artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info(
                f"Data Transformation Artifact : {artifact}"
            )

            return artifact

        except Exception as e:

            logging.exception(e)

            raise NetworkSecurityException(e, sys)