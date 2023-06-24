"""
train_model.py is the wrapper function for the entire
mini-pipeline: getting the data, cleaning training,
evaluating and infering the ML model. Every step is
logged in pipeline.log file
Author: George Christodoulou
Date: 17/06/23

"""
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from settings import settings
from ml.data import process_data

from ml.model import (
    train_model,
    get_model_metrics,
    get_inference,
    get_slices,
    get_confusion_matrix,
    load_pkl,
    save_pkl,
    log_cm,
    log_model,
    log_slice
)


# Initialize logging
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    filemode="a",
    format="%(name)s - %(levelname)s - %(message)s",
)

# Setting the data, target, cat_features,
# model_path, model, encoder and labelizer pkl
data = pd.read_csv(settings["data_path"])
target = settings["target_variable"]
cat_features = settings["cat_features"]
model_save_path = settings["model_path"]
model_pkl = settings["pkl_files"]["model"]
encoder_pkl = settings["pkl_files"]["encoder"]
labelizer_pkl = settings["pkl_files"]["labelizer"]
slices_save_path = settings["slices_save_path"]
# Since labels are imbalanced, I am going to use stratify
# to ensure the distribution of the labels to be the same in
# training and test set.
train, test = train_test_split(
    data, test_size=0.20, random_state=42, stratify=data[target]
)

# Processing the train data with the process_data function. We fit
# the encoder and the labelbinirizer here.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=target, training=True
)

# Process the test data with the process_data function. We use the fitted
# encoder and labelbinirizer to transfor the test set.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)
# Check if a model already exists:
if os.path.isfile(os.path.join(model_save_path, model_pkl)):
    # Load model, encoder and labelizer
    model = load_pkl(model_save_path, model_pkl)
    # Load encoder
    encoder = load_pkl(model_save_path, encoder_pkl)
    # Load labelizer
    lb = load_pkl(model_save_path, labelizer_pkl)
# If not, train and save a model.
else:
    model = train_model(X_train, y_train)
    # save model  to disk in ./model folder
    save_pkl(model, model_save_path, model_pkl)
    save_pkl(encoder, model_save_path, encoder_pkl)
    save_pkl(lb, model_save_path, labelizer_pkl)
    log_model(model_save_path)

# Getting the predictions
y_pred = get_inference(model, X_test)
# Evaluating the model
precision, recall, fbeta = get_model_metrics(y_test, y_pred)
# Calculating confusion matrix
cm = get_confusion_matrix(y_test, y_pred)
log_cm(cm)

# Evaluating the slices of all categorical features
# and saving the results in a csv file and in pipeline log
for feature in cat_features:
    perf_df = get_slices(test, feature, y_test, y_pred)
    perf_df.to_csv(slices_save_path, mode="a", index=False)
    log_slice(feature, perf_df)
