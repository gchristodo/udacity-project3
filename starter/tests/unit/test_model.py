"""
This file has the unit tests for the model
Author: George Christodoulou
Date: 21/06/23
"""

from ...ml.model import get_inference
from ...ml.data import process_data
import pandas as pd
import numpy


def test_load_model(get_model):
    my_model = get_model
    assert my_model is not None


def test_load_encoder(get_encoder):
    my_encoder = get_encoder
    assert my_encoder is not None


def test_load_labelizer(get_lb):
    my_labelizer = get_lb
    assert my_labelizer is not None


def test_dataframe_number_of_columns(get_dataframe):
    data = get_dataframe
    assert data.shape[1] == 15


def test_get_inference(get_model,
                       get_sample,
                       get_encoder,
                       get_lb,
                       get_cat_features):
    sample = pd.DataFrame(get_sample, index=[0])
    sample, _, _, _ = process_data(
                            sample,
                            categorical_features=get_cat_features,
                            training=False,
                            encoder=get_encoder,
                            lb=get_lb
                            )
    my_pred = get_inference(get_model, sample)
    assert isinstance(my_pred[0], numpy.int32)
