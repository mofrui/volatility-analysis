import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from code import util


def save_regression_model(model, name: str = "har_model", subdir: str = "out/harrv"):
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
    HAR-RV uses lagged realized volatility as predictors.

    Adds per-sample inference timing on the test set.

    Parameters:
        df (pd.DataFrame): Feature set from generate_rolling_features, must include 'realized_volatility'.

    Returns:
        model (LinearRegression): Trained HAR-RV model
        test_df (pd.DataFrame): Test set with 'y_pred' and 'inference_time'.
    """
    df = df.copy().sort_values(by=["time_id", "start_time"]).reset_index(drop=True)

    # Create HAR features
    df["rv_lag1"] = df["realized_volatility"].shift(1)
    df["rv_lag5"] = df["realized_volatility"].rolling(5).mean().shift(1)
    df["rv_lag22"] = df["realized_volatility"].rolling(22).mean().shift(1)
    df = df.dropna().reset_index(drop=True)

    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[["rv_lag1", "rv_lag5", "rv_lag22"]].values
    y_train = train_df["realized_volatility"].values
    X_test = test_df[["rv_lag1", "rv_lag5", "rv_lag22"]].values
    y_test = test_df["realized_volatility"].values

    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Inference timing per sample
    inference_times = []
    y_pred = []
    for x in X_test:
        start_t = time.perf_counter()
        pred = model.predict(x.reshape(1, -1))[0]
        inference_times.append(time.perf_counter() - start_t)
        y_pred.append(pred)
    y_pred = np.array(y_pred)

    test_df["y_pred"] = y_pred
    test_df["inference_time"] = inference_times

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    qlike = util.qlike_loss(y_test, y_pred)
    dir_acc = util.directional_accuracy(y_test, y_pred)
    avg_time = np.mean(inference_times)

    print("=== HAR-RV OLS Baseline Evaluation ===")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"QLIKE: {qlike:.4f}")
    print(f"Directional Accuracy: {dir_acc:.4f}")
    print(f"Average inference time per sample: {avg_time:.6f} seconds")

    save_regression_model(model, name="har_ols")
    return model, test_df


def wls(df: pd.DataFrame):
    """
    Train and evaluate HAR-RV model using WLS (weighted least squares).
    Weights = 1 / realized_volatility^2

    Adds per-sample inference timing on the test set.

    Parameters:
        df (pd.DataFrame): Feature set with 'realized_volatility'.

    Returns:
        model: Fitted statsmodels WLS model
        test_df: Test set with 'y_pred' and 'inference_time'.
    """
    df = df.copy().sort_values(by=["time_id", "start_time"]).reset_index(drop=True)

    # Create HAR features
    df["rv_lag1"] = df["realized_volatility"].shift(1)
    df["rv_lag5"] = df["realized_volatility"].rolling(5).mean().shift(1)
    df["rv_lag22"] = df["realized_volatility"].rolling(22).mean().shift(1)
    df = df.dropna().reset_index(drop=True)

    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[["rv_lag1", "rv_lag5", "rv_lag22"]]
    y_train = train_df["realized_volatility"]
    X_test = test_df[["rv_lag1", "rv_lag5", "rv_lag22"]]
    y_test = test_df["realized_volatility"]

    # Statsmodels WLS requires constant
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    weights = 1 / (y_train.values ** 2 + 1e-8)
    model = sm.WLS(y_train, X_train_sm, weights=weights).fit()

    # Inference timing per sample
    inference_times = []
    y_pred = []
    for _, row in X_test_sm.iterrows():
        start_t = time.perf_counter()
        pred = model.predict(row)[0]
        inference_times.append(time.perf_counter() - start_t)
        y_pred.append(pred)
    y_pred = np.array(y_pred)

    test_df["y_pred"] = y_pred
    test_df["inference_time"] = inference_times

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    qlike = util.qlike_loss(y_test.values, y_pred)
    dir_acc = util.directional_accuracy(y_test.values, y_pred)
    avg_time = np.mean(inference_times)

    print("=== HAR-RV WLS Baseline Evaluation ===")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"QLIKE: {qlike:.4f}")
    print(f"Directional Accuracy: {dir_acc:.4f}")
    print(f"Average inference time per sample: {avg_time:.6f} seconds")

    save_regression_model(model, name="har_wls")
    return model, test_df
