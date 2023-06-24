"""
Model.py contains three major functionalities
for training, evaluating and infering the ML model
Author: George Christodoulou
Date: 17/06/23

"""
import multiprocessing
import logging
import json
import os
import pickle
from typing import Tuple, Any
from sklearn.metrics import (fbeta_score,
                             precision_score,
                             recall_score,
                             confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def log_best_params(model):
    """A function to log the best parameters
    after GridSearch

    Args:
        model (?): The trained Model
    """
    logging.info("####### Best parameters #######")
    best_params = model.best_params_
    json_string = json.dumps(best_params)
    logging.info("BEST PARAMS: %s", json_string)


def log_metrics(precision: float, recall: float, fbeta: float):
    """A funciton to log the metrics

    Args:
        precision (float): The precision of the model
        recall (float): The recall of the model
        fbeta (float): The fbeta of the model
    """
    logging.info("precision: %s, recall: %s, fbeta: %s", str(precision), str(recall), str(fbeta))


def log_cm(confusion_matrix: np.ndarray):
    """A funciton to log the confusion matrix

    Args:
        confusion_matrix (np.ndarray): The confusion matrix
    """
    confusion_matrix_string = np.array2string(confusion_matrix, floatmode='unique')
    logging.info("Confusion Matrix: %s", confusion_matrix_string)


def load_pkl(save_path: str, pkl_file: str):
    """A function that loads a pkl file

    Args:
        save_path (str): The path where the pickle is saved
        pkl_file (str): The pickle file

    Returns:
        _type_: The pickle object
    """
    my_pkl = pickle.load(open(os.path.join(save_path, pkl_file), "rb"))
    return my_pkl


def log_model(model_save_path: str):
    """A function that logs the model's savepath

    Args:
        model_save_path (str): The model's savepath
    """
    logging.info("Model saved to disk: %s", model_save_path)


def log_slice(feature: str, perf_df: pd.DataFrame):
    """A function that logs a slice of a feature of
    a pandas dataframe

    Args:
        feature (str): The feature
        perf_df (pd.DataFrame): The dataframe
    """
    logging.info("Performance on slice %s", feature)
    logging.info(perf_df)


def save_pkl(pkl: Any, save_path: str, pkl_file: str):
    """A function that saves a pickle into HDD

    Args:
        pkl (Any): The pickle object
        save_path (str): The Save path
        pkl_file (str): The pickle name
    """
    pickle.dump(pkl, open(os.path.join(save_path, pkl_file), "wb"))


# Optional: implement hyperparameter tuning.
def train_model(X_train: np.ndarray, y_train: np.ndarray):
    """
    This function trains a model with RandomForest,
    using Gridsearch to find the optimal parameters
    and it returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Label data.
    Returns
    -------
    model
        Trained RF model.
    """
    param_grid = {
        "n_estimators": [100, 200, 300],  # Number of trees in the forest
        "max_depth": [None, 5, 10],  # Maximum depth of the trees
        "min_samples_split": [
            2,
            5,
            10,
        ],  # Minimum number of samples required to split an internal node
        "min_samples_leaf": [
            1,
            2,
            4,
        ],  # Minimum number of samples required to be at a leaf node
        "max_features": [
            "sqrt",
            "log2",
        ],  # Number of features to consider at each split
        "bootstrap": [True, False],  # Whether to bootstrap samples
    }
    njobs = multiprocessing.cpu_count() - 1
    logging.info("Searching best hyperparameters... Using %d cores", njobs)

    rf_clf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        n_jobs=njobs,
    )

    rf_clf.fit(X_train, y_train)
    log_best_params(rf_clf)

    return rf_clf


def get_model_metrics(
    y: np.ndarray, preds: np.ndarray
) -> Tuple[float, float, float]:
    """
    A function that calculates precision, recall, and F1 for the
    evalution of the RF model.

    Inputs
    ------
    y : np.ndarray
        Known labels.
    preds : np.ndarray
        Predicted labels.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    log_metrics(precision, recall, fbeta)
    return precision, recall, fbeta


def get_inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    y_pred = model.predict(X)
    return y_pred


def get_slices(
    df: pd.DataFrame, feature: str, y: np.ndarray, preds: np.ndarray
) -> pd.DataFrame:
    """
    A function that clculates the performance metrics
    for each unique value of a given categorical feature.
    ------
    df:
        The preprocessed test dataframe.
    feature:
        The feature we slice.
    y : np.ndarray
        The Labels.
    preds : np.ndarray
        The predicted labels.

    Returns
    ------
    Dataframe with
        feature: str -The categorical feature name.
        feature_value: str - The category name.
        n_samples: integer - number of data samples in that category.
        precision : float
        recall : float
        fbeta : float
    """

    slice_options = df[feature].unique().tolist()
    perf_df = pd.DataFrame(
        {
            "feature": feature,
            "n_samples": [],
            "precision": [],
            "recall": [],
            "fbeta": [],
        }
    )
    for option in slice_options:
        slice_mask = df[feature] == option
        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]
        precision, recall, fbeta = get_model_metrics(slice_y, slice_preds)
        perf_df.loc[option] = {
            "feature": feature,
            "n_samples": len(slice_y),
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        }

    perf_df.reset_index(inplace=True)
    perf_df.columns = ["feature", "feature_value", *perf_df.columns[2:]]
    return perf_df


def get_confusion_matrix(y: np.ndarray, preds: np.ndarray):
    """
    A function that produces the confusion matrix
    ------
    y : np.array
        Known labels
    preds : np.array
        Predicted labels
    Returns
    ------
    cm : The confusion matrix
    """
    cm = confusion_matrix(y, preds)
    return cm
