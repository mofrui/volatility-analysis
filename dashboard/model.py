# model.py (cleaned)
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model as keras_load_model

# # --- Load Model ---
# def load_model(model_path, custom_objects=None):
#     return keras_load_model(model_path, compile=False)


def load_model(model_path, custom_objects=None):
    return keras_load_model(model_path, compile=False)

# --- Load Preprocessed Data ---
def prepare_lstm_data(stock_id, time_id):
    file_path = f"dashboard/data/{stock_id}_tid{time_id}.pkl"
    df = pd.read_pickle(file_path)

    X = np.stack(df["X"].values)
    y = df["y"].values
    time_ids = df["time_id"].values
    start_times = df["start_time"].values if "start_time" in df else np.arange(len(y))

    return X, y, time_ids, start_times

# --- Evaluation Metrics ---

def qlike_loss(y_true, y_pred):
    var_true = y_true ** 2
    var_pred = y_pred ** 2
    return ((var_true / var_pred - np.log(var_true / var_pred) - 1).mean()).item()

def directional_accuracy(y_true, y_pred):
    actual_change = np.diff(y_true)
    predicted_change = np.diff(y_pred)
    return (actual_change * predicted_change > 0).mean().item()

def mse_custom(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2).item()

def rmse_custom(y_true, y_pred):
    return np.sqrt(mse_custom(y_true, y_pred))

