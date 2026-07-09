import yaml
from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging
import os, sys
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score  # better for classification

def read_yaml_file(file_path: str) -> dict:
    """Reads a YAML file and returns its contents as a dictionary."""
    try:
        with open(file_path, 'r') as yaml_file:  # text mode
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def save_numpy_array_data(file_path: str, array: np.array):
    """Save numpy array data to file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_object(file_path: str, obj: object) -> None:
    try: 
        logging.info("Entered the save_object method of mainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object of mainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file:{file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """Load numpy array data from file."""
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    """Evaluate models with GridSearchCV and return test scores."""
    try:
        report = {}
        for model_name, model in models.items():
            param_grid = params.get(model_name, {})
            gs = GridSearchCV(
                 estimator=model,
                param_grid=param_grid,
                 cv=3,
                scoring="accuracy",
                 n_jobs=1,
                 )
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)  # fixed typo
            model.fit(x_train, y_train)

            y_test_pred = model.predict(x_test)
            test_model_score = accuracy_score(y_test, y_test_pred)  # classification metric

            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys)
