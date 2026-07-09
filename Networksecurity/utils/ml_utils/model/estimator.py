import sys
import pandas as pd

from Networksecurity.exception.exception import NetworkSecurityException


class NetworkModel:

    def __init__(self, preprocessor, model):
        """
        Wrapper class that combines the preprocessor and the trained model.
        """

        try:

            self.preprocessor = preprocessor
            self.model = model

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, x):

        try:

            # -------------------------------
            # Convert to DataFrame if needed
            # -------------------------------

            if not isinstance(x, pd.DataFrame):
                x = pd.DataFrame(x)

            # -------------------------------
            # Feature Validation
            # -------------------------------

            if hasattr(self.preprocessor, "feature_names_in_"):

                expected_columns = list(
                    self.preprocessor.feature_names_in_
                )

                missing_columns = list(
                    set(expected_columns) - set(x.columns)
                )

                if len(missing_columns) > 0:
                    raise Exception(
                        f"Missing columns: {missing_columns}"
                    )

                extra_columns = list(
                    set(x.columns) - set(expected_columns)
                )

                if len(extra_columns) > 0:
                    x = x.drop(
                        columns=extra_columns,
                        errors="ignore",
                    )

                # Reorder columns exactly as during training
                x = x[expected_columns]

            # -------------------------------
            # Data Transformation
            # -------------------------------

            transformed_data = self.preprocessor.transform(x)

            # -------------------------------
            # Prediction
            # -------------------------------

            prediction = self.model.predict(
                transformed_data
            )

            return prediction

        except Exception as e:
            raise NetworkSecurityException(e, sys)