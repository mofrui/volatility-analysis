import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


# Length of the rolling window used to extract historical features
WINDOW_SIZE = 330

# Number of future time steps to forecast for each window
FORECAST_HORIZON = 10

#  Step size for moving the rolling window forward
STEP = 5

def create_snapshot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute static snapshot features from raw order book data.
    These features are computed per timestamp before rolling window processing.

    Parameters:
        df (pd.DataFrame): Raw order book snapshot data.

    Returns:
        pd.DataFrame: DataFrame with engineered snapshot-level features.
    """
    df = df.copy()

    # Mid price and WAP
    df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2
    df['wap'] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (
        df['bid_size1'] + df['ask_size1'] + 1e-6
    )

    # Spread and relative spread
    df['bid_ask_spread'] = df['ask_price1'] - df['bid_price1']
    df['spread_pct'] = df['bid_ask_spread'] / (df['mid_price'] + 1e-6)

    # Order book imbalance and depth
    df['imbalance'] = (df['bid_size1'] - df['ask_size1']) / (df['bid_size1'] + df['ask_size1'] + 1e-6)
    df['depth_ratio'] = df['bid_size1'] / (df['ask_size1'] + 1e-6)

    # Log return of WAP
    df['log_return'] = np.log(df['wap']).diff().fillna(0)

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


def generate_tick_sequences(df: pd.DataFrame, feature_cols: List[str], window: int = 330, horizon: int = 10, step: int = 5) -> pd.DataFrame:
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the snapshot-level order book data.
    This includes log changes, rolling statistics, and normalized spread features.
    
    Parameters:
        df (pd.DataFrame): DataFrame with at least ['time_id', 'seconds_in_bucket', 'wap', 'log_return', 'spread_pct', etc.]

    Returns:
        pd.DataFrame: DataFrame with additional feature columns.
    """
    df = df.copy()

    # Log change in WAP (alternative to log_return)
    df['log_wap_change'] = np.log(df['wap'] / df['wap'].shift(1)).replace([np.inf, -np.inf], 0).fillna(0)

    # Rolling std of log return (over 5-second window within each time_id)
    df['rolling_std_logret'] = (
        df.groupby('time_id')['log_return']
        .rolling(window=10, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    # Rolling mean and std of spread_pct for z-score calculation
    grouped = df.groupby('time_id')['spread_pct']
    spread_mean = grouped.transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    spread_std = grouped.transform(lambda x: x.rolling(window=10, min_periods=1).std().replace(0, np.nan))
    df['spread_zscore'] = ((df['spread_pct'] - spread_mean) / spread_std).fillna(0)

    # Volume imbalance (alternative to imbalance ratio)
    df['volume_imbalance'] = df['ask_size1'] - df['bid_size1']

    # Clip any extreme values (for safety)
    for col in ['log_wap_change', 'rolling_std_logret', 'spread_zscore', 'volume_imbalance']:
        df[col] = df[col].clip(-10, 10)

    return df


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
