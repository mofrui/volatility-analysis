import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
import statsmodels.api as sm
from typing import Tuple
from itertools import product
from tqdm import tqdm
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error
import pickle
import os

from code import util

warnings.filterwarnings("ignore")


def save_garch_model(model, name="baseline", subdir="models/out/garch"):
    """
    Save fitted arch_model result object (ARCHModelResult) to a .pkl file.

    Parameters:
        model: Fitted ARCH model result (e.g. from arch_model().fit()).
        name (str): Filename without extension.
        subdir (str): Directory to save model into.
    """
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ GARCH model saved to {path}")


def baseline(rolling_df: pd.DataFrame, full_df: pd.DataFrame, scale: int = 1000000, train_ratio: float = 0.8):
    """
    ARMA(1,1)-GARCH(1,1) baseline with internal train/test split (by time_id) and tqdm progress bar.

    Parameters:
        rolling_df (pd.DataFrame): DataFrame with realized_volatility, start_time, time_id.
        full_df (pd.DataFrame): Full snapshot-level df with 'log_return' and 'time_id'.
        scale (int): Scaling factor for log returns. Default = 10000.
        train_ratio (float): Ratio of time_ids to use for training. Default = 0.8

    Returns:
        model_summary: Last fitted model (for inspection)
        result_df: Test set with [time_id, start_time, y_true, y_pred]
    """
    W, H = 330, 10
    results = []
    last_model = None

    # --- Step 1: train/test split by time_id ---
    unique_time_ids = rolling_df["time_id"].unique()
    split_index = int(len(unique_time_ids) * train_ratio)
    train_ids = unique_time_ids[:split_index]
    test_ids = unique_time_ids[split_index:]

    test_df = rolling_df[rolling_df["time_id"].isin(test_ids)].reset_index(drop=True)

    # --- Step 2: loop over test set with progress bar ---
    print("Running ARMA-GARCH forecast...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        time_id = row["time_id"]
        start_time = row["start_time"]
        y_true = row["realized_volatility"]

        sub_df = full_df[full_df["time_id"] == time_id]
        sub_df = sub_df[sub_df["seconds_in_bucket"] <= start_time].copy()

        if len(sub_df) < W:
            continue

        train_returns = sub_df.iloc[-W:]["log_return"].values * scale

        try:
            model = arch_model(train_returns, mean="ARX", lags=1, vol="GARCH", p=1, q=1)
            res = model.fit(disp="off")
            last_model = res

            forecast = res.forecast(horizon=H)
            pred_var = forecast.variance.values[-1]
            pred_vol = np.sqrt(np.sum(pred_var)) / scale

            results.append({
                "time_id": time_id,
                "start_time": start_time,
                "y_true": y_true,
                "y_pred": pred_vol
            })

        except Exception as e:
            print(f"[Warning] GARCH failed at time_id {time_id}, start_time {start_time}: {e}")
            continue

    result_df = pd.DataFrame(results)

    # --- Step 3: Evaluation ---
    y_true = result_df["y_true"].values
    y_pred = result_df["y_pred"].values
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    qlike = util.qlike_loss(y_true, y_pred)
    directional_acc = util.directional_accuracy(y_true, y_pred)

    print("=== ARMA-GARCH Baseline Evaluation (on test set) ===")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"QLIKE: {qlike:.4f}")
    print(f"Directional Accuracy: {directional_acc:.4f}")
    print(f"Success Rate: {len(result_df)}/{len(test_df)}")

    save_garch_model(last_model, name="baseline")

    return last_model, result_df




def baseline_grid(rolling_df: pd.DataFrame, full_df: pd.DataFrame, 
                  scale: int = 1000000, train_ratio: float = 0.8,
                  p_values=[1, 2], q_values=[1, 2]) -> tuple:
    """
    Try multiple (p, q) GARCH parameter combinations and select the best based on RMSE.

    Returns:
        best_model: The best GARCH model object
        best_result_df: Corresponding result dataframe with columns [time_id, start_time, y_true, y_pred]
    """
    W, H = 330, 10
    unique_time_ids = rolling_df["time_id"].unique()
    split_index = int(len(unique_time_ids) * train_ratio)
    test_ids = unique_time_ids[split_index:]
    test_df = rolling_df[rolling_df["time_id"].isin(test_ids)].reset_index(drop=True)

    best_rmse = float("inf")
    best_model = None
    best_result_df = None

    print(f"🔍 Grid search over (p, q) combinations: {list(product(p_values, q_values))}\n")

    for p, q in product(p_values, q_values):
        result_list = []
        last_model = None

        print(f"▶ Evaluating GARCH({p},{q})")
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"GARCH({p},{q})"):
            time_id = row["time_id"]
            start_time = row["start_time"]
            y_true = row["realized_volatility"]

            sub_df = full_df[full_df["time_id"] == time_id]
            sub_df = sub_df[sub_df["seconds_in_bucket"] <= start_time].copy()

            if len(sub_df) < W:
                continue

            train_returns = sub_df.iloc[-W:]["log_return"].values * scale

            try:
                model = arch_model(train_returns, mean="ARX", lags=1, vol="GARCH", p=p, q=q)
                res = model.fit(disp="off")
                last_model = res

                forecast = res.forecast(horizon=H)
                pred_var = forecast.variance.values[-1]
                pred_vol = np.sqrt(np.sum(pred_var)) / scale

                result_list.append({
                    "time_id": time_id,
                    "start_time": start_time,
                    "y_true": y_true,
                    "y_pred": pred_vol
                })

            except Exception:
                continue

        result_df = pd.DataFrame(result_list)

        if result_df.empty:
            continue

        # --- Step 3: Evaluation ---
        y_true = result_df["y_true"].values
        y_pred = result_df["y_pred"].values
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        qlike = util.qlike_loss(y_true, y_pred)
        directional_acc = util.directional_accuracy(y_true, y_pred)

        print(f"    → RMSE = {rmse:.8f}, QLIKE = {qlike:.4f}, Directional Accuracy = {directional_acc:.4f}\n")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = last_model
            best_result_df = result_df

    print("✅ Grid Search Complete. Best Model Summary:")
    print(best_model.summary())
    print(f"Best Model: → RMSE = {rmse:.8f}, QLIKE = {qlike:.4f}, Directional Accuracy = {directional_acc:.4f}\n")
    print(f"garch_best_model_p{best_model.model.p}_q{best_model.model.q}")


    name = f"garch_best_model_p{best_model.model.p}_q{best_model.model.q}"
    save_garch_model(best_model, name=name)


    return best_model, best_result_df



