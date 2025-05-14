import pandas as pd
import numpy as np
import joblib
import os
import random

# Set random seed
np.random.seed(3888)
random.seed(3888)


# Load the model file
def load_model(model_file):
    model = joblib.load(model_file)
    return model

#Calculate the prediction
def mse(true, pred):
    return np.mean((true - pred)**2)

def qlike(true, pred):
    return np.mean(np.log(pred**2) + (true**2) / (pred**2))

def rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))

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