import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import pickle
import time
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
    These features are computed per timestamp before rolling window processing.

    Parameters:
        df (pd.DataFrame): Raw order book snapshot data.

    Returns:
        pd.DataFrame: DataFrame with engineered snapshot-level features.
    """
    df = df.copy()

    # Mid price and WAP
    df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2
    df['wap'] = (
        df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    ) / (df['bid_size1'] + df['ask_size1'] + 1e-6)

    # Spread and relative spread
    df['bid_ask_spread'] = df['ask_price1'] - df['bid_price1']
    df['spread_pct'] = df['bid_ask_spread'] / (df['mid_price'] + 1e-6)

    # Order book imbalance and depth
    df['imbalance'] = (
        df['bid_size1'] - df['ask_size1']
    ) / (df['bid_size1'] + df['ask_size1'] + 1e-6)
    df['depth_ratio'] = df['bid_size1'] / (df['ask_size1'] + 1e-6)

    # Log return of WAP
    df['log_return'] = np.log(df['wap']).diff().fillna(0)

    # Log change in WAP (alternative to log_return)
    df['log_wap_change'] = (
        np.log(df['wap'] / df['wap'].shift(1))
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Rolling standard deviation of log return within each time_id
    df['rolling_std_logret'] = (
        df.groupby('time_id')['log_return']
          .rolling(window=10, min_periods=1)
          .std()
          .reset_index(level=0, drop=True)
          .fillna(0)
    )

    # Rolling z-score of spread percentage
    spread_mean = df.groupby('time_id')['spread_pct']\
                    .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    spread_std = df.groupby('time_id')['spread_pct']\
                    .transform(lambda x: x.rolling(window=10, min_periods=1).std().replace(0, np.nan))
    df['spread_zscore'] = ((df['spread_pct'] - spread_mean) / spread_std).fillna(0)

    # Volume imbalance between ask and bid
    df['volume_imbalance'] = df['ask_size1'] - df['bid_size1']

    # Clip extreme values for safety
    for col in ['log_wap_change', 'rolling_std_logret', 'spread_zscore', 'volume_imbalance']:
        df[col] = df[col].clip(-10, 10)

    return df


def generate_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate rolling-window statistical features and realized volatility labels.

    Parameters:
        df (pd.DataFrame): DataFrame containing engineered snapshot features.

    Returns:
        pd.DataFrame: Rolling window features with realized volatility label.
    """
    df = df.copy()
    feature_list = []

    for idx in range(0, len(df) - WINDOW_SIZE - FORECAST_HORIZON, STEP):
        window = df.iloc[idx:idx + WINDOW_SIZE]
        future = df.iloc[idx + WINDOW_SIZE : idx + WINDOW_SIZE + FORECAST_HORIZON]

        features = {
            'stock_id': df.iloc[idx]['stock_id'],
            'time_id': df.iloc[idx]['time_id'],
            'start_time': df.iloc[idx]['seconds_in_bucket']
        }

        # Aggregated stats over rolling window
        for col in ['wap', 'spread_pct', 'imbalance', 'depth_ratio', 'log_return']:
            features[f'{col}_mean'] = window[col].mean()
            features[f'{col}_std'] = window[col].std()
            features[f'{col}_max'] = window[col].max()
            features[f'{col}_min'] = window[col].min()

        # Future realized volatility label
        future_returns = future['log_return'].values
        realized_vol = np.sqrt(np.sum(future_returns ** 2))
        features['realized_volatility'] = realized_vol

        feature_list.append(features)

    return pd.DataFrame(feature_list)


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


def plot_prediction_vs_actual(df: pd.DataFrame, time_id: int, y_true_col: str = "y_true", y_pred_col: str = "y_pred"):
    """
    Plot predicted vs. actual realized volatility over time for a given time_id.

    Parameters:
        df (pd.DataFrame): DataFrame from LSTM test result, with columns [time_id, start_time, y_true, y_pred]
        time_id (int): Time ID to filter for plotting
        y_true_col (str): Column name for true values
        y_pred_col (str): Column name for predicted values
    """
    df_plot = df[df["time_id"] == time_id].copy()

    if df_plot.empty:
        print(f"[Warning] No data found for time_id = {time_id}")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["start_time"], df_plot[y_true_col], label="Actual Volatility", linewidth=2)
    plt.plot(df_plot["start_time"], df_plot[y_pred_col], label="Predicted Volatility", linewidth=2, linestyle="--")
    plt.title(f"Prediction vs Actual (time_id = {time_id})")
    plt.xlabel("Seconds in Bucket")
    plt.ylabel("Realized Volatility")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


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


def plot_inference_time_boxplot(val_df: pd.DataFrame, inference_col: str = 'inference_time'):
    """
    Plot a boxplot of inference time per rolling window during validation.
    """
    if inference_col not in val_df:
        print(f"[Warning] Column '{inference_col}' not found in DataFrame.")
        return
    times = val_df[inference_col].dropna()
    plt.figure(figsize=(8, 6))
    plt.boxplot(times, vert=True, showfliers=True)
    plt.title("Validation Inference Time per Window")
    plt.ylabel("Time (seconds)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def plot_side_by_side_inference_boxplots(val_dfs: dict,
                                         inference_col: str = 'inference_time',
                                         percentile: float = 95.0,
                                         figsize: tuple = (10, 6)):
    filtered_data = []
    labels = []

    for name, df in val_dfs.items():
        if inference_col not in df:
            raise ValueError(f"DataFrame for '{name}' does not have '{inference_col}'")
        times = df[inference_col].dropna().values
        thresh = np.percentile(times, percentile)
        filtered = times[times <= thresh]
        filtered_data.append(filtered)
        labels.append(name)

    plt.figure(figsize=figsize)
    plt.boxplot(filtered_data, labels=labels, showfliers=False)
    plt.title(f'Inference Time per Window (≤ {percentile}th percentile)')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



def out_of_sample_evaluation(model_path: str,
                             scaler_path: str,
                             snapshot_df: pd.DataFrame,
                             feature_cols: list):
    """
    Out-of-sample evaluation with per-window inference timing.

    Returns:
      val_df: DataFrame with columns
        ['time_id','start_time','y_true','y_pred','inference_time']
      metrics: dict with mse, rmse, qlike, directional_accuracy, avg_inference_time
    """
    # 1. load model & scalers
    model = load_model(model_path, compile=False)
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    x_scaler = scalers['x_scaler']
    y_scaler = scalers['y_scaler']

    # 2. generate sequences
    seq_df = generate_tick_sequences(snapshot_df, feature_cols=feature_cols)
    X = np.stack(seq_df['X'].values)
    y_true = seq_df['y'].values
    time_ids = seq_df['time_id'].values
    starts = seq_df['start_time'].values

    # 3. scale X
    n_windows, W, D = X.shape
    X_scaled = x_scaler.transform(X.reshape(-1, D)).reshape(n_windows, W, D)

    # 4. predict with timing
    inference_times = []
    preds_s = []
    for window in X_scaled:
        t0 = time.perf_counter()
        p_s = model.predict(window[np.newaxis, ...], verbose=0).ravel()[0]
        inference_times.append(time.perf_counter() - t0)
        preds_s.append(p_s)
    preds_s = np.array(preds_s)
    y_pred = y_scaler.inverse_transform(preds_s.reshape(-1, 1)).ravel()

    # 5. assemble val_df
    val_df = pd.DataFrame({
        'time_id':        time_ids,
        'start_time':     starts,
        'y_true':         y_true,
        'y_pred':         y_pred,
        'inference_time': inference_times
    })

    # 6. compute metrics
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    qlike = qlike_loss(y_true, y_pred)
    dir_acc = directional_accuracy(y_true, y_pred)
    avg_time = np.mean(inference_times)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'qlike': qlike,
        'directional_accuracy': dir_acc,
        'avg_inference_time': avg_time
    }

    return val_df, metrics

def compute_metrics_by_time_id(val_df: pd.DataFrame):
    """
    Compute MSE, RMSE, QLIKE, directional accuracy, and avg inference time for each time_id.
    Returns a DataFrame indexed by time_id with columns [mse, rmse, qlike, dir_acc, avg_inf_time].
    """
    records = []
    for tid, group in val_df.groupby('time_id'):
        y_true = group['y_true'].values
        y_pred = group['y_pred'].values
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        ql = qlike_loss(y_true, y_pred)
        da = directional_accuracy(y_true, y_pred)
        avg_time = group['inference_time'].mean() if 'inference_time' in group else np.nan
        records.append({
            'time_id': tid,
            'mse': mse,
            'rmse': rmse,
            'qlike': ql,
            'dir_acc': da,
            'avg_inf_time': avg_time
        })
    return pd.DataFrame(records).set_index('time_id')
    

def plot_metrics_boxplot(metrics_df: pd.DataFrame):
    """
    Plot boxplots for each evaluation metric across time_ids.
    Expects a DataFrame with columns ['mse','rmse','qlike','dir_acc'] and time_id index.
    """
    plt.figure(figsize=(12, 8))
    metrics = ['mse','rmse','qlike','dir_acc']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.boxplot(metrics_df[metric].dropna(), vert=True, showfliers=True)
        plt.title(metric.upper())
        plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
