# spread_model.py
import pandas as pd
import joblib
import os


def load_precomputed_features(csv_path="dashboard/data/predictions_spy.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    return df


def load_spread_model(model_path="dashboard/Models/bid_ask_spread_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found.")
    return joblib.load(model_path)
