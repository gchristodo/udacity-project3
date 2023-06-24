"""
This file has the unit tests for the API
Author: George Christodoulou
Date: 21/06/23
"""
from fastapi.testclient import TestClient
from settings import settings
from main import app
import json


# Instantiate the testing client with the app
client = TestClient(app)

# Testing get method
def test_api_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to Udacity 3rd Project"


# Testing case: Prediction >50K
def test_inference_for_prediction_1():
    sample = settings["sample"]
    data = json.dumps(sample)
    print("THIS IS DATA: ", data)
    r = client.post("/inference/", data=data)
    # Testing response
    assert r.status_code == 200
    # Parsing response content as JSON
    r_json = json.loads(r.content)
    print("This is the response: ", r_json)
    # Testing random features from sample
    assert r_json["sex"] == "Male"
    assert r_json["occupation"] == "Protective-serv"
    # Testing prediction
    assert r_json["prediction"] == ">50K"


# Testing case: Prediction <=50K
def test_inference_for_prediction_0():
    sample = settings["sample_2"]
    data = json.dumps(sample)
    print("THIS IS DATA: ", data)
    r = client.post("/inference/", data=data)
    # Testing response
    assert r.status_code == 200
    # Parsing response content as JSON
    r_json = json.loads(r.content)
    print("This is the response: ", r_json)
    # Testing random features from sample
    assert r_json["sex"] == "Male"
    assert r_json["occupation"] == "Sales"
    # Testing prediction
    assert r_json["prediction"] == "<=50K"


# Testing case: Wrong Input JSON
def test_wrong_query_input():
    sample = settings["sample_3"]
    data = json.dumps(sample)
    r = client.post("/inference/", data=data)
    r_json = json.loads(r.content)
    assert "prediction" not in r_json.keys()