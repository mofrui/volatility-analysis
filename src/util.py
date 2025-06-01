import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import pickle
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Length of the rolling window used to extract historical features
WINDOW_SIZE = 330

# Number of future time steps to forecast for each window
FORECAST_HORIZON = 10

# Step size for moving the rolling window forward
STEP = 5

def create_snapshot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute snapshot-level features from raw order book data.
    These features are computed per timestamp before any rolling-window aggregation.

    Parameters:
        df (pd.DataFrame): Raw order book snapshot data.

    Returns:
        pd.DataFrame: DataFrame augmented with engineered snapshot-level features.
    """
    df = df.copy()

    # 1. Mid-price: average of best bid and best ask
    df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2

    # 2. Volume-weighted average price (VWAP) using best levels
    df['wap'] = (
        df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    ) / (df['bid_size1'] + df['ask_size1'] + 1e-6)

    # 3. Absolute and relative spread
    df['spread'] = df['ask_price1'] - df['bid_price1']
    df['spread_pct'] = df['spread'] / (df['mid_price'] + 1e-6)

    # 4. Order book imbalance and depth ratio at the top level
    df['imbalance'] = (
        df['bid_size1'] - df['ask_size1']
    ) / (df['bid_size1'] + df['ask_size1'] + 1e-6)
    df['depth_ratio'] = df['bid_size1'] / (df['ask_size1'] + 1e-6)

    # 5. Log return of VWAP relative to previous snapshot
    df['log_return'] = np.log(df['wap']).diff().fillna(0)

    # 6. Log ratio change of VWAP as alternative measure
    df['log_wap_change'] = (
        np.log(df['wap'] / df['wap'].shift(1))
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # 7. Rolling standard deviation of the log returns per time_id
    df['rolling_std_logret'] = (
        df.groupby('time_id')['log_wap_change']
          .rolling(window=10, min_periods=1)
          .std()
          .reset_index(level=0, drop=True)
          .fillna(0)
    )

    # 8. Rolling z-score of spread percentage per time_id
    spread_mean = df.groupby('time_id')['spread_pct']\
                    .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    spread_std = df.groupby('time_id')['spread_pct']\
                    .transform(lambda x: x.rolling(window=10, min_periods=1).std().replace(0, np.nan))
    df['spread_zscore'] = ((df['spread_pct'] - spread_mean) / spread_std).fillna(0)

    # 9. Volume imbalance: difference between ask and bid sizes
    df['volume_imbalance'] = df['ask_size1'] - df['bid_size1']

    # 10. Clip extreme feature values to avoid outliers
    cols_to_clip = [
        'log_wap_change',
        'rolling_std_logret',
        'spread_zscore',
        'volume_imbalance'
    ]
    for col in cols_to_clip:
        df[col] = df[col].clip(-10, 10)

    return df


def generate_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling-window summary statistics and realized volatility labels using global parameters.

    Uses:
        WINDOW_SIZE       : length of historical window (number of snapshots)
        FORECAST_HORIZON  : number of future snapshots for volatility label
        STEP              : step size to advance the rolling window

    Parameters:
        df (pd.DataFrame): DataFrame of snapshot-level features, sorted by time.

    Returns:
        pd.DataFrame: Each row contains:
            - stock_id, time_id, window_start_sec
            - For each feature (<feat>): <feat>_mean, _std, _max, _min
            - realized_volatility: sqrt of sum of squared future returns
    """
    # Work on a copy to avoid modifying the original
    df = df.copy().reset_index(drop=True)
    records = []

    # Define the maximum valid start index for rolling windows
    max_start_idx = len(df) - WINDOW_SIZE - FORECAST_HORIZON

    # Slide the rolling window by STEP until max_start_idx
    for start_idx in range(0, max_start_idx + 1, STEP):
        # Historical window slice
        window_df = df.iloc[start_idx : start_idx + WINDOW_SIZE]
        # Future period slice for label calculation
        future_df = df.iloc[start_idx + WINDOW_SIZE : start_idx + WINDOW_SIZE + FORECAST_HORIZON]

        # Base identifiers: stock_id, time_id, and starting second of the window
        rec = {
            'stock_id': window_df.iloc[0]['stock_id'],
            'time_id': window_df.iloc[0]['time_id'],
            'start_time': window_df.iloc[0]['seconds_in_bucket']
        }

        # List of snapshot-level features to summarize
        stat_features = ['wap', 'spread_pct', 'imbalance', 'depth_ratio', 'log_return']
        for feat in stat_features:
            vals = window_df[feat]
            rec[f'{feat}_mean'] = vals.mean()
            rec[f'{feat}_std']  = vals.std()
            rec[f'{feat}_max']  = vals.max()
            rec[f'{feat}_min']  = vals.min()

        # Compute realized volatility from future log returns
        future_returns = future_df['log_return'].to_numpy()
        rec['realized_volatility'] = np.sqrt((future_returns ** 2).sum())

        records.append(rec)

    # Assemble records into DataFrame
    return pd.DataFrame(records)


def generate_tick_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: int = WINDOW_SIZE,
    horizon: int = FORECAST_HORIZON,
    step: int = STEP
) -> pd.DataFrame:
    """
    Generate rolling tick sequences without crossing time_id to avoid data leakage.

    Parameters:
        df (pd.DataFrame): Input dataframe with features, log_return, time_id, and seconds_in_bucket.
        feature_cols (list): List of feature column names to include in X.
        window (int): Size of the rolling window (W).
        horizon (int): Forecast horizon (H).
        step (int): Step size for rolling window (S).

    Returns:
        pd.DataFrame: A DataFrame with columns [X, y, time_id, start_time]
    """
    df = df.copy()
    X_list, y_list, time_ids, start_times = [], [], [], []

    for time_id, group in df.groupby("time_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        for i in range(0, n - window - horizon + 1, step):
            X_seq = group.iloc[i:i+window][feature_cols].values
            future_returns = group.iloc[i+window:i+window+horizon]["log_return"].values
            realized_vol = np.sqrt(np.sum(future_returns ** 2))
            X_list.append(X_seq)
            y_list.append(realized_vol)
            time_ids.append(time_id)
            start_times.append(group.iloc[i]["seconds_in_bucket"])

    return pd.DataFrame({
        "X": X_list,
        "y": y_list,
        "time_id": time_ids,
        "start_time": start_times
    })


def qlike_loss(y_true, y_pred):
    """
    QLIKE Loss: penalizes underestimation of volatility more than overestimation.
    """
    var_true = y_true ** 2
    var_pred = y_pred ** 2
    return np.mean(var_true / var_pred - np.log(var_true / var_pred) - 1)


def directional_accuracy(y_true, y_pred):
    """
    Computes directional accuracy: how often the direction (up/down) was predicted correctly.
    """
    actual_change = np.diff(y_true)
    predicted_change = np.diff(y_pred)
    return np.mean((actual_change * predicted_change) > 0)


def plot_training_loss(history):
    """
    Plot training and validation loss over epochs for LTSM model.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("LSTM Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_prediction_vs_actual_multi(
    val_dfs_vis: dict,
    time_id: int,
    start_time: int = None,
    end_time: int = None,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred"
):
    """
    Plot predicted vs. actual realized volatility over time for a given time_id
    across multiple models.

    Parameters:
        val_dfs_vis (dict): Dictionary with model names as keys and DataFrames as values.
                            Each DataFrame should have columns [time_id, start_time, y_true, y_pred].
        time_id (int): Time ID to filter for plotting.
        start_time (int): Optional lower bound of start_time to zoom in.
        end_time (int): Optional upper bound of start_time to zoom in.
        y_true_col (str): Column name for true values.
        y_pred_col (str): Column name for predicted values.
    """
    plt.figure(figsize=(12, 5))

    for label, df in val_dfs_vis.items():
        df_plot = df[df["time_id"] == time_id].copy()
        if start_time is not None:
            df_plot = df_plot[df_plot["start_time"] >= start_time]
        if end_time is not None:
            df_plot = df_plot[df_plot["start_time"] <= end_time]

        if df_plot.empty:
            continue

        plt.plot(df_plot["start_time"], df_plot[y_pred_col], label=f"{label} (Predicted)", linestyle="--")
        if label == list(val_dfs_vis.keys())[0]:
            # Plot actual only once
            plt.plot(df_plot["start_time"], df_plot[y_true_col], label="Actual Volatility", linewidth=2)

    plt.title(f"Prediction vs Actual for Time ID {time_id} (Zoom: {start_time}~{end_time})")
    plt.xlabel("Seconds in Bucket")
    plt.ylabel("Realized Volatility")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_rmse_robustness(val_dfs: dict, figsize=(8, 4)):
    records = []
    for name, df in val_dfs.items():
        agg = (
            df.groupby('time_id')
              .apply(lambda g: pd.Series({
                  'rmse': np.sqrt(np.mean((g['y_pred'] - g['y_true'])**2))
              }))
              .reset_index()
        )
        agg['model'] = name
        records.append(agg)
    metrics_df = pd.concat(records, ignore_index=True)
    
    models = list(val_dfs.keys())
    data   = [metrics_df.loc[metrics_df.model==m, 'rmse'].values for m in models]
    
    fig, ax = plt.subplots(figsize=figsize)
    bxp = ax.boxplot(
        data,
        labels=models,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True
    )
    
    for box in bxp['boxes']:
        box.set_facecolor('#A0CBE8')
        box.set_alpha(0.7)
    for median in bxp['medians']:
        median.set_color('orange')
        median.set_linewidth(2)
    for line in bxp['whiskers'] + bxp['caps']:
        line.set_color('gray')
        line.set_linewidth(1.5)


    
    ax.set_title('RMSE Robustness for Evaluating Models')
    ax.set_ylabel('RMSE')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.show()


def create_evaluation_metrics_table(val_dfs: dict) -> pd.DataFrame:
    """
    Create a summary table of evaluation metrics (RMSE, MSE, QLIKE, Inference Time)
    across all models (DataFrames in val_dfs).

    Parameters:
        val_dfs (dict): Dictionary with model names as keys and DataFrames as values.

    Returns:
        pd.DataFrame: Summary table with mean values per model.
    """
    records = []
    for name, df in val_dfs.items():
        grouped = df.groupby('time_id').apply(
            lambda g: pd.Series({
                'rmse': np.sqrt(np.mean((g['y_pred'] - g['y_true'])**2)),
                'mse': np.mean((g['y_pred'] - g['y_true'])**2),
                'qlike': np.mean(
                    (g['y_true']**2) / (g['y_pred']**2 + 1e-8) -
                    np.log((g['y_true']**2) / (g['y_pred']**2 + 1e-8)) - 1
                ),
                'inference_time': g['inference_time'].mean() if 'inference_time' in g.columns else np.nan
            })
        ).reset_index()

        # Take mean of metrics across time_ids for each model
        mean_metrics = grouped[['rmse', 'mse', 'qlike', 'inference_time']].mean()
        mean_metrics['model'] = name
        records.append(mean_metrics)

    # Create final summary DataFrame
    metrics_table = pd.DataFrame(records).set_index('model')
    return metrics_table



def out_of_sample_evaluation(
    model_path: str,
    scaler_path: str,
    snapshot_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, float]]:

    model = load_model(model_path, compile=False)
    if isinstance(model, list):
        model = model[-1]

    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    x_scaler = scalers['x_scaler']
    y_scaler = scalers['y_scaler']

    seq_df = generate_tick_sequences(snapshot_df, feature_cols=feature_cols)
    X_windows = np.stack(seq_df['X'].to_numpy())
    y_true    = seq_df['y'].to_numpy()
    time_ids  = seq_df['time_id'].to_numpy()
    start_ts  = seq_df['start_time'].to_numpy()

    n_samples, window_len, n_feats = X_windows.shape
    X_flat = X_windows.reshape(-1, n_feats)
    X_scaled = x_scaler.transform(X_flat).reshape(n_samples, window_len, n_feats)

    inference_times = []
    raw_preds = []

    for window in X_scaled:
        t0 = time.perf_counter()
        out = model.predict(window[np.newaxis, ...], verbose=0)
        inference_times.append(time.perf_counter() - t0)

        pred_val = out[0].ravel()[0] if isinstance(out, list) else out.ravel()[0]
        raw_preds.append(pred_val)

    y_pred = y_scaler.inverse_transform(np.array(raw_preds).reshape(-1, 1)).ravel()

    val_df = pd.DataFrame({
        'time_id':        time_ids,
        'start_time':     start_ts,
        'y_true':         y_true,
        'y_pred':         y_pred,
        'inference_time': inference_times
    })

    return val_df
