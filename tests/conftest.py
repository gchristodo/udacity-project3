"""
This file has the conftest file for the unittests
Author: George Christodoulou
Date: 21/06/23
"""
from pytest import fixture
from settings import settings
import pandas as pd
from ml.model import load_pkl


@fixture(scope="session")
def get_dataframe():
    path = settings["data_path"]
    data = pd.read_csv(path)
    return data


@fixture(scope="session")
def get_model():
    model_save_path = settings["model_path"]
    model_pkl = settings["pkl_files"]["model"]
    try:
        my_model = load_pkl(model_save_path, model_pkl)
    except FileNotFoundError:
        my_model = None
    return my_model


@fixture(scope="session")
def get_encoder():
    model_save_path = settings["model_path"]
    encoder_pkl = settings["pkl_files"]["encoder"]
    try:
        my_encoder = load_pkl(model_save_path, encoder_pkl)
    except FileNotFoundError:
        my_encoder = None
    return my_encoder


@fixture(scope="session")
def get_lb():
    model_save_path = settings["model_path"]
    labelizer_pkl = settings["pkl_files"]["labelizer"]
    try:
        my_lb = load_pkl(model_save_path, labelizer_pkl)
    except FileNotFoundError:
        my_lb = None
    return my_lb


@fixture(scope="session")
def get_sample():
    my_sample = settings["sample"]
    return my_sample


@fixture(scope="session")
def get_cat_features():
    cat_features = settings["cat_features"]
    return cat_features
