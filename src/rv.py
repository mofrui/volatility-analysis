import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from src import util


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

    # 1. 构造 HAR 特征
    df["rv_lag1"]  = df["realized_volatility"].shift(1)
    df["rv_lag5"]  = df["realized_volatility"].rolling(5).mean().shift(1)
    df["rv_lag22"] = df["realized_volatility"].rolling(22).mean().shift(1)
    df = df.dropna().reset_index(drop=True)

    # 2. 按 time_id 做 80/20 划分
    unique_ids = sorted(df["time_id"].unique())
    split_idx  = int(len(unique_ids) * 0.8)
    train_ids, test_ids = unique_ids[:split_idx], unique_ids[split_idx:]

    train_df = df[df["time_id"].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df["time_id"].isin(test_ids)].reset_index(drop=True)

    # 3. 准备 X/y
    X_train = train_df[["rv_lag1", "rv_lag5", "rv_lag22"]].values
    y_train = train_df["realized_volatility"].values
    X_test  = test_df[ ["rv_lag1", "rv_lag5", "rv_lag22"]].values
    y_test  = test_df["realized_volatility"].values

    # 4. 训练模型
    model = LinearRegression().fit(X_train, y_train)

    # 5. 测试集逐样本预测并计时
    inference_times, y_pred = [], []
    for x in X_test:
        t0 = time.perf_counter()
        pred = model.predict(x.reshape(1, -1))[0]
        inference_times.append(time.perf_counter() - t0)
        y_pred.append(pred)
    test_df["y_pred"]        = y_pred
    test_df["inference_time"]= inference_times
    

    # 6. 评估
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

    # 7. 保存模型
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
