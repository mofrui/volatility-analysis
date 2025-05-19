import pandas as pd
import numpy as np
import joblib
import os
import random
import pickle
from keras.saving import register_keras_serializable
import tensorflow as tf
import keras.backend as K

# Set random seed
np.random.seed(3888)
random.seed(3888)


# LSTM 

# Load the LSTM model (Keras)
from tensorflow.keras.models import load_model as keras_load_model



def load_model(model_path, custom_objects=None):
    return keras_load_model(model_path, custom_objects=custom_objects)

# Load preprocessed test data per stock
def prepare_lstm_data(stock_id, time_id):
    file_path = f"dashboard/data/dashboard_lstm_{stock_id}_tid{time_id}.pkl"
    df = pd.read_pickle(file_path)

    X = np.stack(df["X"].values)
    y = df["y"].values
    time_ids = df["time_id"].values
    start_times = df["start_time"].values if "start_time" in df else np.arange(len(y))

    return X, y, time_ids, start_times


# LSTM 
# # Load the model file
# def load_model(model_file):
#     model = joblib.load(model_file)
#     return model

@register_keras_serializable()
def qlike_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.log(tf.square(y_pred)) + tf.square(y_true) / tf.square(y_pred))


@register_keras_serializable()
def directional_accuracy(y_true, y_pred):
    direction_true = tf.math.sign(y_true[1:] - y_true[:-1])
    direction_pred = tf.math.sign(y_pred[1:] - y_pred[:-1])
    return tf.reduce_mean(tf.cast(tf.equal(direction_true, direction_pred), tf.float32))

@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

@register_keras_serializable()
def rmse(mse):
    return np.sqrt(mse)
    

# Preprocess the data
def compute_orderbook_features(df):
    df = df.copy()
    df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2
    df['wap'] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['ask_size1'] + df['bid_size1'])
    df['bid_ask_spread'] = df['ask_price1'] - df['bid_price1']
    df['spread_pct'] = df['bid_ask_spread'] / df['mid_price']
    df['spread_variation'] = df.groupby('time_id')['spread_pct'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
    df['imbalance'] = (df['bid_size1'] - df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    df['depth_ratio'] = df['bid_size1'] / df['ask_size1'].replace(0, np.nan)
    return df

# Load the data
def load_data(stock_id):
    # Paths
    folder_path = "Data"
    feature_path = os.path.join(folder_path, "order_book_feature.parquet")
    target_path = os.path.join(folder_path, "order_book_target.parquet")

    # Load feature/target metadata (only to list available stocks)
    feature_df = pd.read_parquet(feature_path, engine='pyarrow')
    target_df = pd.read_parquet(target_path, engine='pyarrow')

    # Load per-stock data
    stock_feature = feature_df[feature_df["stock_id"] == int(stock_id)].copy()
    stock_target = target_df[target_df["stock_id"] == int(stock_id)].copy()
    stock_df = pd.concat([stock_feature, stock_target], axis=0).sort_values(['time_id', 'seconds_in_bucket'])
    stock_df = compute_orderbook_features(stock_df)

    # Compute realized volatility
    stock_df["log_return"] = stock_df.groupby("time_id")["wap"].transform(lambda x: np.log(x / x.shift(1)))
    rv_df = stock_df.groupby("time_id")["log_return"].agg(lambda x: np.sqrt(np.sum(x ** 2))).reset_index()
    rv_df = rv_df.rename(columns={"log_return": "realized_volatility"})

    # Create lag features
    hav_df = rv_df.copy()
    hav_df["rv_lag_1"] = hav_df["realized_volatility"].shift(1)
    hav_df["rv_lag_5"] = hav_df["realized_volatility"].shift(5)
    hav_df["rv_lag_10"] = hav_df["realized_volatility"].shift(10)
    hav_df = hav_df.dropna().reset_index(drop=True)

    return hav_df