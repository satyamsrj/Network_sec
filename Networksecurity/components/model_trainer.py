import os
import sys

import dagshub
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)

from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging

from Networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from Networksecurity.entity.config_entity import (
    ModelTrainerConfig,
)

from Networksecurity.utils.ml_utils.model.estimator import (
    NetworkModel,
)

from Networksecurity.utils.ml_utils.metric.classification import (
    get_classification_score,
)

from Networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)

# ------------------------------------------------------------
# Initialize DagsHub + MLflow
# ------------------------------------------------------------

dagshub.init(
    repo_owner="satyamsrj",
    repo_name="Network_sec",
    mlflow=True,
)


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):

        try:

            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # --------------------------------------------------------
    # MLflow Logger
    # --------------------------------------------------------

    def track_mlflow(
        self,
        best_model,
        train_metric,
        test_metric,
    ):

        try:

            with mlflow.start_run():

                # Train Metrics

                mlflow.log_metric(
                    "train_f1_score",
                    train_metric.f1_score,
                )

                mlflow.log_metric(
                    "train_precision",
                    train_metric.precision_score,
                )

                mlflow.log_metric(
                    "train_recall",
                    train_metric.recall_score,
                )

                # Test Metrics

                mlflow.log_metric(
                    "test_f1_score",
                    test_metric.f1_score,
                )

                mlflow.log_metric(
                    "test_precision",
                    test_metric.precision_score,
                )

                mlflow.log_metric(
                    "test_recall",
                    test_metric.recall_score,
                )

                # Save model in MLflow

                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    name="model",
                )

                logging.info("MLflow logging completed.")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # --------------------------------------------------------
    # Train Models
    # --------------------------------------------------------

    def train_model(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
    ):

        try:

            logging.info("Starting model training...")

            models = {

                "Random Forest": RandomForestClassifier(
                    random_state=42,
                    verbose=1,
                ),

                "Decision Tree": DecisionTreeClassifier(
                    random_state=42,
                ),

                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42,
                    verbose=1,
                ),

                "Logistic Regression": LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                ),

                "AdaBoost": AdaBoostClassifier(
                    random_state=42,
                ),

            }

            params = {

                "Decision Tree": {

                    "criterion": [
                        "gini",
                        "entropy",
                        "log_loss",
                    ]

                },

                "Random Forest": {

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                    ]

                },

                "Gradient Boosting": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001,
                    ],

                    "subsample": [
                        0.6,
                        0.7,
                        0.75,
                        0.8,
                        0.85,
                        0.9,
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                    ],

                },

                "Logistic Regression": {},

                "AdaBoost": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001,
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                    ],

                },

            }

            logging.info("Evaluating models...")

            model_report = evaluate_models(

                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params,

            )

            logging.info(f"Model Report : {model_report}")

            best_model_name = max(
                model_report,
                key=model_report.get,
            )

            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            logging.info(
                f"Best Model : {best_model_name}"
            )

            logging.info(
                f"Best Score : {best_model_score}"
            )

            
             # ---------------------------------------------------
            # Train Best Model
            # ---------------------------------------------------

            logging.info("Training best model...")

            best_model.fit(x_train, y_train)

            # ---------------------------------------------------
            # Predictions
            # ---------------------------------------------------

            y_train_pred = best_model.predict(x_train)

            train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred,
            )

            y_test_pred = best_model.predict(x_test)

            test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred,
            )

            logging.info(f"Train Metrics : {train_metric}")
            logging.info(f"Test Metrics : {test_metric}")

            # ---------------------------------------------------
            # MLflow Logging
            # ---------------------------------------------------

            self.track_mlflow(
                best_model=best_model,
                train_metric=train_metric,
                test_metric=test_metric,
            )

            # ---------------------------------------------------
            # Load Preprocessor
            # ---------------------------------------------------

            logging.info("Loading preprocessor...")

            preprocessor = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # ---------------------------------------------------
            # Create Network Model
            # ---------------------------------------------------

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model,
            )

            # ---------------------------------------------------
            # Save Artifact Model
            # ---------------------------------------------------

            model_dir = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )

            os.makedirs(
                model_dir,
                exist_ok=True,
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=network_model,
            )

            logging.info(
                f"Artifact Model Saved : {self.model_trainer_config.trained_model_file_path}"
            )

            # ---------------------------------------------------
            # Save Standalone Model
            # ---------------------------------------------------

            os.makedirs(
                "final_models",
                exist_ok=True,
            )

            save_object(
                file_path="final_models/model.pkl",
                obj=best_model,
            )

            logging.info(
                "Standalone model saved : final_models/model.pkl"
            )

            # ---------------------------------------------------
            # Create Artifact
            # ---------------------------------------------------

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )

            logging.info(
                f"Model Trainer Artifact : {model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:

            logging.exception(e)

            raise NetworkSecurityException(e, sys)

    # --------------------------------------------------------
    # Initiate Model Trainer
    # --------------------------------------------------------

    def initiate_model_trainer(
        self,
    ) -> ModelTrainerArtifact:

        try:

            logging.info("Loading transformed train and test arrays...")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            logging.info(
                f"x_train Shape : {x_train.shape}"
            )

            logging.info(
                f"x_test Shape : {x_test.shape}"
            )

            model_trainer_artifact = self.train_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )

            logging.info(
                "Model training completed successfully."
            )

            return model_trainer_artifact

        except Exception as e:

            logging.exception(e)

            raise NetworkSecurityException(e, sys)