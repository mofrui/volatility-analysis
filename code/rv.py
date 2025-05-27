import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from code import util


def save_regression_model(model, name: str = "har_model", subdir: str = "out/rv"):
    """
    Save sklearn or statsmodels regression model using pickle.

    Parameters:
        model: Trained regression model (LinearRegression or statsmodels result).
        name (str): Filename (without extension).
        subdir (str): Folder path to save the model in.
    """
    os.makedirs(subdir, exist_ok=True)
    path = f"{subdir}/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {path}")


def ols(df: pd.DataFrame):
    """
    Train and evaluate HAR-RV model on rolling volatility features.
    Splitting is done by time_id (80% train / 20% test) to avoid leakage.
    """
    df = df.copy().sort_values(by=["time_id", "start_time"]).reset_index(drop=True)

    df["rv_lag1"]  = df["realized_volatility"].shift(1)
    df["rv_lag5"]  = df["realized_volatility"].rolling(5).mean().shift(1)
    df["rv_lag22"] = df["realized_volatility"].rolling(22).mean().shift(1)
    df = df.dropna().reset_index(drop=True)

    unique_ids = sorted(df["time_id"].unique())
    split_idx  = int(len(unique_ids) * 0.8)
    train_ids, test_ids = unique_ids[:split_idx], unique_ids[split_idx:]

    train_df = df[df["time_id"].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df["time_id"].isin(test_ids)].reset_index(drop=True)

    X_train = train_df[["rv_lag1", "rv_lag5", "rv_lag22"]].values
    y_train = train_df["realized_volatility"].values
    X_test  = test_df[ ["rv_lag1", "rv_lag5", "rv_lag22"]].values
    y_test  = test_df["realized_volatility"].values

    model = LinearRegression().fit(X_train, y_train)

    inference_times, y_pred = [], []
    for x in X_test:
        t0 = time.perf_counter()
        pred = model.predict(x.reshape(1, -1))[0]
        inference_times.append(time.perf_counter() - t0)
        y_pred.append(pred)
    test_df["y_pred"]        = y_pred
    test_df["inference_time"]= inference_times
    
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    qlike = util.qlike_loss(y_test, np.array(y_pred))
    da   = util.directional_accuracy(y_test, y_pred)
    avgt = np.mean(inference_times)

    print("=== HAR-RV OLS Baseline Evaluation ===")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"QLIKE: {qlike:.4f}")
    print(f"Directional Accuracy: {da:.4f}")
    print(f"Average inference time per sample: {avgt:.6f} seconds")

    save_regression_model(model, name="rv_ols")

    val_df = test_df[[
        'time_id',
        'start_time',
        'realized_volatility',
        'y_pred',
        'inference_time'
    ]].rename(columns={
        'realized_volatility': 'y_true'
    })

    return model, val_df


def wls(df: pd.DataFrame):
    """
    Train & evaluate HAR-RV using WLS with 80/20 time_id split,
    returning the same val_df format as ols().
    """
    df = df.copy().sort_values(by=["time_id","start_time"]).reset_index(drop=True)
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
    print("=== HAR-RV WLS Baseline Evaluation ===")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"QLIKE: {qlike:.4f}")
    print(f"Directional Accuracy: {da:.4f}")

    save_regression_model(model, name="har_wls")

    return model, val_df