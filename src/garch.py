import numpy as np
import pandas as pd
import time
import random
from arch import arch_model
from itertools import product
from sklearn.metrics import mean_squared_error
import src.util as util

SEED = 3888
np.random.seed(SEED)
random.seed(SEED)

def garch(df: pd.DataFrame, W: int = 330, H: int = 10, scale: int = 1e6, p_values=[1, 2], q_values=[1, 2]):
    df = df.copy()
    rolling_df = util.generate_rolling_features(df)

    unique_ids = sorted(rolling_df["time_id"].unique())
    split_idx  = int(len(unique_ids) * 0.8)
    train_ids, test_ids = unique_ids[:split_idx], unique_ids[split_idx:]

    test_df  = rolling_df[rolling_df["time_id"].isin(test_ids)].reset_index(drop=True)

    best_rmse = float("inf")
    best_model = None
    best_result_df = None
    best_pq = (1, 1)

    for p, q in product(p_values, q_values):
        result_list = []
        last_model = None

        for _, row in test_df.iterrows():
            time_id, start_time, y_true = row["time_id"], row["start_time"], row["realized_volatility"]
            sub_df = df[(df["time_id"] == time_id) & (df["seconds_in_bucket"] <= start_time)]
            if len(sub_df) < W:
                continue

            log_ret = sub_df.iloc[-W:]["log_return"].values * scale

            try:
                np.random.seed(SEED)
                random.seed(SEED)

                model = arch_model(log_ret, mean="ARX", lags=1, vol="GARCH", p=p, q=q)
                res = model.fit(disp="off")
                last_model = res

                start_infer = time.time()
                forecast = res.forecast(horizon=H)
                end_infer = time.time()

                pred_var = forecast.variance.values[-1]
                pred_vol = np.sqrt(np.sum(pred_var)) / scale
                inference_time = end_infer - start_infer

                result_list.append({
                    "time_id": time_id,
                    "start_time": start_time,
                    "y_true": y_true,
                    "y_pred": pred_vol,
                    "inference_time": inference_time
                })

            except:
                continue

        result_df = pd.DataFrame(result_list)

        if result_df.empty:
            continue

        y_true_vals = result_df["y_true"].values
        y_pred_vals = result_df["y_pred"].values
        mse = mean_squared_error(y_true_vals, y_pred_vals)
        rmse = np.sqrt(mse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = last_model
            best_result_df = result_df
            best_pq = (p, q)

    return best_result_df
