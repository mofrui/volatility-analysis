import os
import pandas as pd
import numpy as np
from pathlib import Path
import models.util as util

# Stock IDs used in dashboard
stock_ids = [50200, 104919, 22771]

# Parquet data sources
df_feat = pd.read_parquet("data/order_book_feature.parquet")
df_tgt = pd.read_parquet("data/order_book_target.parquet")

# Features used for LSTM
feature_cols = [
    "wap", "spread_pct", "imbalance", "depth_ratio", "log_return",
    "log_wap_change", "rolling_std_logret", "spread_zscore", "volume_imbalance"
]
# feature_cols = ["wap", "log_return"]


for stock_id in stock_ids:
    pkl_path = f"data/preprocessed_9_{stock_id}.pkl"
    if Path(pkl_path).exists():
        print(f"✅ {pkl_path} already exists. Skipping...")
        continue

    print(f"⏳ Processing stock {stock_id}...")
    df_feat_stock = df_feat[df_feat["stock_id"] == stock_id].copy()
    df_tgt_stock = df_tgt[df_tgt["stock_id"] == stock_id].copy()

    df = pd.concat([df_feat_stock, df_tgt_stock], axis=0)
    df = df.sort_values(by=["stock_id", "time_id", "seconds_in_bucket"]).reset_index(drop=True)

    df = util.create_snapshot_features(df)
    df = util.add_features(df)

    valid_ids = df.groupby("time_id").filter(lambda g: len(g) >= 340)["time_id"].unique()
    # limited_ids = valid_ids[:100] 
    # df = df[df["time_id"].isin(limited_ids)].copy()
    df = df[df["time_id"].isin(valid_ids)].copy()


    lstm_df = util.generate_tick_sequences(df, feature_cols)
    if lstm_df.empty:
        print(f"⚠️ No valid LSTM data for stock {stock_id}. Skipping...")
        continue

    # Extract the last timestep from each sequence (shape: [samples, features])
    X_array = np.stack(lstm_df["X"].values)[:, -1, :]  # shape = (N, 9)
    X = pd.DataFrame(X_array, columns=feature_cols)
    X["realized_volatility"] = lstm_df["y"].values


    X.to_pickle(pkl_path)
    print(f"✅ Saved {pkl_path}")

print("🎉 Precomputation complete.")
