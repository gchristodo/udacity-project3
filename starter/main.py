"""
This is the code for creating the FastAPI.
Author: George Christodoulou
Date: 20/06/23
"""
# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from ml.model import load_pkl
from ml.data import process_data
from settings import settings


model = None
encoder = None
lb = None


# This is the input data object along with
# its elements and their respective data types.
class Input(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
                "example": {
                            "age": 42,
                            "workclass": "Local-gov",
                            "fnlgt": 201495,
                            "education": "Some-college",
                            "education_num": 10,
                            "marital_status": "Married-civ-spouse",
                            "occupation": "Protective-serv",
                            "relationship": "Husband",
                            "race": "White",
                            "sex": "Male",
                            "capital_gain": 0,
                            "capital_loss": 0,
                            "hours_per_week": 72,
                            "native_country": "United-States"
                }
        }


app = FastAPI(title="Predict Income API",
              description="This is an API that takes a person with specific features and predicts if his/her annual salary is greater than 50K")


# Loading pickles when init for better efficiency.
@app.on_event("startup")
async def initiate():
    global model, encoder, lb
    try:
        model_save_path = settings["model_path"]
        model_pkl = settings["pkl_files"]["model"]
        encoder_pkl = settings["pkl_files"]["encoder"]
        labelizer_pkl = settings["pkl_files"]["labelizer"]
        # Load model, encoder and labelizer
        model = load_pkl(model_save_path, model_pkl)
        # Load encoder
        encoder = load_pkl(model_save_path, encoder_pkl)
        # Load labelizer
        lb = load_pkl(model_save_path, labelizer_pkl)
    except FileNotFoundError:
        error_message = "Model, encoder or labelizer pickles not found"
        print(error_message)


# GET that gives a greeting
@app.get("/")
async def welcome():
    return "Welcome to Udacity 3rd Project"


# POST that sends our sample to our API and produces the inference
@app.post("/inference/")
async def get_inference(sample: Input):
    global model, encoder, lb
    if not model or not encoder or not lb:
        try:
            model_save_path = settings["model_path"]
            model_pkl = settings["pkl_files"]["model"]
            encoder_pkl = settings["pkl_files"]["encoder"]
            labelizer_pkl = settings["pkl_files"]["labelizer"]
            # Load model, encoder and labelizer
            model = load_pkl(model_save_path, model_pkl)
            # Load encoder
            encoder = load_pkl(model_save_path, encoder_pkl)
            # Load labelizer
            lb = load_pkl(model_save_path, labelizer_pkl)
        except FileNotFoundError:
            error_message = "Model, encoder or labelizer pickles not found"
            print(error_message)
    data = {
                "age": sample.age,
                "workclass": sample.workclass,
                "fnlgt": sample.fnlgt,
                "education": sample.education,
                "education_num": sample.education_num,
                "marital_status": sample.marital_status,
                "occupation": sample.occupation,
                "relationship": sample.relationship,
                "race": sample.race,
                "sex": sample.sex,
                "capital_gain": sample.capital_gain,
                "capital_loss": sample.capital_loss,
                "hours_per_week": sample.hours_per_week,
                "native_country": sample.native_country
            }
    # Transforming the sample to a single dataframe-row for processing
    sample = pd.DataFrame(data, index=[0])
    # Processing sample
    sample, _, _, _ = process_data(
                            sample,
                            categorical_features=settings["cat_features"],
                            training=False,
                            encoder=encoder,
                            lb=lb
                            )
    # Get the inference
    prediction = model.predict(sample)
    # Output of prediction is a 1D array either 1 for >50K or 0 for <=50K
    if prediction[0] == 1:
        prediction = '>50K'
    else:
        prediction = '<=50K'
    data["prediction"] = prediction

    return data

if __name__ == '__main__':
    pass
