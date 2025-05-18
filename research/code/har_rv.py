import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import pickle
import os

from code import util

def save_regression_model(model, name: str = "har_model", subdir: str = "models/out/harrv"):
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

    Parameters:
        df (pd.DataFrame): Feature set from generate_rolling_features, must include 'realized_volatility'

    Returns:
        model (LinearRegression): Trained HAR-RV model
        test_df (pd.DataFrame): Test set with predictions added as 'y_pred'
    """
    df = df.copy().reset_index(drop=True)

    # Sort to preserve time order
    df = df.sort_values(by=["time_id", "start_time"]).reset_index(drop=True)

    # Create HAR features: daily, weekly, monthly realized volatility (rolling mean of target)
    df["rv_lag1"] = df["realized_volatility"].shift(1)
    df["rv_lag5"] = df["realized_volatility"].rolling(5).mean().shift(1)
    df["rv_lag22"] = df["realized_volatility"].rolling(22).mean().shift(1)

    df = df.dropna().reset_index(drop=True)

    # 80-20 time-based split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[["rv_lag1", "rv_lag5", "rv_lag22"]]
    y_train = train_df["realized_volatility"]
    X_test = test_df[["rv_lag1", "rv_lag5", "rv_lag22"]]
    y_test = test_df["realized_volatility"]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    test_df["y_pred"] = y_pred

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    qlike = util.qlike_loss(y_test.values, y_pred)
    directional_acc = util.directional_accuracy(y_test.values, y_pred)

    print("=== HAR-RV OLS Baseline Evaluation ===")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"QLIKE: {qlike:.4f}")
    print(f"Directional Accuracy: {directional_acc:.4f}")

    save_regression_model(model, name="har_ols")

    return model, test_df


def wls(df: pd.DataFrame):
    """
    Train and evaluate HAR-RV model using WLS (weighted least squares).
    Weights = 1 / realized_volatility^2

    Parameters:
        df (pd.DataFrame): Feature set with 'realized_volatility'

    Returns:
        model: Fitted statsmodels WLS model
        test_df: Test set with predictions in 'y_pred'
    """
    df = df.copy().reset_index(drop=True)
    df = df.sort_values(by=["time_id", "start_time"]).reset_index(drop=True)

    # Create HAR lags
    df["rv_lag1"] = df["realized_volatility"].shift(1)
    df["rv_lag5"] = df["realized_volatility"].rolling(5).mean().shift(1)
    df["rv_lag22"] = df["realized_volatility"].rolling(22).mean().shift(1)
    df = df.dropna().reset_index(drop=True)

    # 80-20 split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[["rv_lag1", "rv_lag5", "rv_lag22"]]
    y_train = train_df["realized_volatility"]
    X_test = test_df[["rv_lag1", "rv_lag5", "rv_lag22"]]
    y_test = test_df["realized_volatility"]

    # Add intercept for statsmodels
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Define weights (inverse volatility squared)
    weights = 1 / (y_train**2 + 1e-8)

    # Train WLS model
    model = sm.WLS(y_train, X_train, weights=weights).fit()

    # Predict
    y_pred = model.predict(X_test)
    test_df["y_pred"] = y_pred

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    qlike = util.qlike_loss(y_test.values, y_pred)
    directional_acc = util.directional_accuracy(y_test.values, y_pred)

    print("=== HAR-RV WLS Baseline Evaluation ===")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"QLIKE: {qlike:.4f}")
    print(f"Directional Accuracy: {directional_acc:.4f}")

    save_regression_model(model, name="har_wls")

    return model, test_df
