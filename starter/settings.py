"""
settings.py contains the settings used in the
training_model.py
Author: George Christodoulou
Date: 17/06/23

"""

settings = {
    "data_path": "data/census_cleaned.csv",
    "slices_save_path": "data/slices.csv",
    "target_variable": "salary",
    "cat_features": [
                        "workclass",
                        "education",
                        "marital_status",
                        "occupation",
                        "relationship",
                        "race",
                        "sex",
                        "native_country",
                    ],
    "model_path": "model",
    "pkl_files": {
        "model": "rfc.pkl",
        "encoder": "encoder.pkl",
        "labelizer": "labelizer.pkl"
    },
    "sample": {
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
    },
    "sample_2": {
                    'age': 38,
                    'workclass': "Private",
                    'fnlgt': 28887,
                    'education': "11th",
                    'education_num': 7,
                    'marital_status': "Married-civ-spouse",
                    'occupation': "Sales",
                    'relationship': "Husband",
                    'race': "White",
                    'sex': "Male",
                    'capital_gain': 0,
                    'capital_loss': 0,
                    'hours_per_week': 50,
                    'native_country': "United-States"
    },
    "sample_3": {
                    'age': 38,
                    'workclass': "Private",
                    'fnlgt': 28887
    },

}