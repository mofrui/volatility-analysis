import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

import src.util as util

def wls(df: pd.DataFrame):
    """
    Train & evaluate HAR-RV using WLS with 80/20 time_id split,
    returning the same val_df format as ols().
    """
    df = df.copy()
    df = util.generate_rolling_features(df)
    df = df.sort_values(by=["time_id","start_time"]).reset_index(drop=True)
    df["rv_lag1"]  = df["realized_volatility"].shift(1)
    df["rv_lag5"]  = df["realized_volatility"].rolling(5).mean().shift(1)
    df["rv_lag22"] = df["realized_volatility"].rolling(22).mean().shift(1)
    df = df.dropna().reset_index(drop=True)

    unique_ids = sorted(df["time_id"].unique())
    split_idx  = int(len(unique_ids) * 0.8)
    train_ids, test_ids = unique_ids[:split_idx], unique_ids[split_idx:]

    train_df = df[df["time_id"].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df["time_id"].isin(test_ids)].reset_index(drop=True)

    time_ids_out  = test_df["time_id"].values
    starts_out    = test_df["start_time"].values
    y_test        = test_df["realized_volatility"].values

    X_train = train_df[["rv_lag1","rv_lag5","rv_lag22"]]
    y_train = train_df["realized_volatility"]
    X_test  = test_df[ ["rv_lag1","rv_lag5","rv_lag22"]]
    X_train_const = sm.add_constant(X_train)
    X_test_const  = sm.add_constant(X_test)

    weights = 1.0/(y_train**2 + 1e-8)
    model   = sm.WLS(y_train, X_train_const, weights=weights).fit()

    inference_times = []
    y_pred = []
    for i in range(len(X_test_const)):
        row = X_test_const.iloc[[i]]
        t0  = time.perf_counter()
        pred = model.predict(row).values[0]
        inference_times.append(time.perf_counter() - t0)
        y_pred.append(pred)
    y_pred = np.array(y_pred)

    val_df = pd.DataFrame({
        "time_id":        time_ids_out,
        "start_time":     starts_out,
        "y_true":         y_test,
        "y_pred":         y_pred,
        "inference_time": inference_times
    })

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    qlike = util.qlike_loss(y_test, y_pred)
    da = util.directional_accuracy(y_test, y_pred)

    return model, val_df