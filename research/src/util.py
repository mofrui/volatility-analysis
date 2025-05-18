import numpy as np
import pandas as pd

# Enhanced snapshot feature generation with Level 1/2 and advanced microstructure indicators

def create_snapshot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detailed snapshot-level features from raw order book data, including:
      - Level 1 & 2 price/size metrics
      - Micro-price, slopes, imbalances
      - Temporal encodings and short-term momentum
    """
    df = df.copy()

    # --- Level 1 metrics ---
    df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2
    df['wap'] = (
        df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    ) / (df['bid_size1'] + df['ask_size1'] + 1e-6)
    df['spread_pct'] = (df['ask_price1'] - df['bid_price1']) / (df['mid_price'] + 1e-6)
    df['imbalance'] = (df['bid_size1'] - df['ask_size1']) / (df['bid_size1'] + df['ask_size1'] + 1e-6)
    df['depth_ratio'] = df['bid_size1'] / (df['ask_size1'] + 1e-6)

    # --- Level 2 metrics ---
    df['mid_price2'] = (df['bid_price2'] + df['ask_price2']) / 2
    df['wap2'] = (
        df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    ) / (df['bid_size2'] + df['ask_size2'] + 1e-6)
    df['spread2_pct'] = (df['ask_price2'] - df['bid_price2']) / (df['mid_price2'] + 1e-6)
    df['imbalance2'] = (df['bid_size2'] - df['ask_size2']) / (df['bid_size2'] + df['ask_size2'] + 1e-6)
    df['depth_ratio2'] = df['bid_size2'] / (df['ask_size2'] + 1e-6)

    # --- Microstructure derived features ---
    # Micro-price (volume-weighted mid)
    df['micro_price'] = (
        df['ask_price1'] * df['bid_size1'] + df['bid_price1'] * df['ask_size1']
    ) / (df['bid_size1'] + df['ask_size1'] + 1e-6)
    # Order-book slope between levels
    df['slope_ask'] = (df['ask_price2'] - df['ask_price1']) / (df['ask_size2'] + 1e-6)
    df['slope_bid'] = (df['bid_price1'] - df['bid_price2']) / (df['bid_size2'] + 1e-6)
    # Cumulative depth imbalance across levels
    df['cum_depth_imbalance'] = (
        (df['bid_size1'] + df['bid_size2']) - (df['ask_size1'] + df['ask_size2'])
    ) / ((df['bid_size1'] + df['bid_size2'] + df['ask_size1'] + df['ask_size2']) + 1e-6)
    # Spread slope ratio
    df['spread_slope'] = df['spread2_pct'] / (df['spread_pct'] + 1e-6)

    # --- Temporal & momentum features ---
    # Log return of Level 1 WAP and short-term momentum
    df['log_return'] = np.log(df['wap']).diff().fillna(0)
    df['momentum_5'] = (
        df.groupby('time_id')['log_return']
          .rolling(window=5, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )
    # Mid-price return
    df['mid_return'] = df.groupby('time_id')['mid_price'].pct_change().fillna(0)

    # --- Cyclical time encoding ---
    max_sec = df['seconds_in_bucket'].max() + 1
    df['sin_time'] = np.sin(2 * np.pi * df['seconds_in_bucket'] / max_sec)
    df['cos_time'] = np.cos(2 * np.pi * df['seconds_in_bucket'] / max_sec)

    # --- Order-flow imbalance (deltas) ---
    df['delta_bid1'] = df.groupby('time_id')['bid_size1'].diff().fillna(0)
    df['delta_ask1'] = df.groupby('time_id')['ask_size1'].diff().fillna(0)
    df['delta_bid2'] = df.groupby('time_id')['bid_size2'].diff().fillna(0)
    df['delta_ask2'] = df.groupby('time_id')['ask_size2'].diff().fillna(0)

    return df


def generate_rolling_features(
    df: pd.DataFrame,
    feature_cols: list = None,
    window: int = 330,
    horizon: int = 10,
    step: int = 5
):
    """
    Generate rolling-window statistical features and a realized volatility label.

    Parameters:
        df (pd.DataFrame): DataFrame of snapshot-level features, including 'log_return'.
        feature_cols (list): Columns to aggregate; defaults to all numeric features except IDs.
        window (int): Size of the rolling window.
        horizon (int): Number of future steps for label.
        step (int): Step size to move the window.

    Returns:
        pd.DataFrame: Each row has aggregated stats for one window and 'realized_volatility'.
    """
    df = df.copy().reset_index(drop=True)
    # Determine features
    exclude = {'stock_id', 'time_id', 'seconds_in_bucket'}
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

    records = []
    for i in range(0, len(df) - window - horizon + 1, step):
        win = df.iloc[i:i+window]
        fut = df.iloc[i+window:i+window+horizon]
        rec = {
            'time_id': win.iloc[0]['time_id'],
            'start_time': win.iloc[0]['seconds_in_bucket']
        }
        # Aggregate stats
        for col in feature_cols:
            vals = win[col]
            rec[f'{col}_mean'] = vals.mean()
            rec[f'{col}_std'] = vals.std()
            rec[f'{col}_max'] = vals.max()
            rec[f'{col}_min'] = vals.min()
        # Label
        lr = fut['log_return'].values
        rec['realized_volatility'] = np.sqrt((lr**2).sum())
        records.append(rec)
    return pd.DataFrame(records)



def generate_tick_sequences(
    df: pd.DataFrame,
    feature_cols: list = None,
    window: int = 330,
    horizon: int = 10,
    step: int = 5
) -> pd.DataFrame:
    """
    Generate rolling tick sequences without crossing time_id, using a flexible feature set.
    Pass `feature_cols` to select a subset; defaults to all engineered features.
    """
    df = df.copy()

    # Collect all engineered features as default
    default_features = [
        # Level 1 & 2
        'wap', 'spread_pct', 'imbalance', 'depth_ratio',
        'mid_price2', 'wap2', 'spread2_pct', 'imbalance2', 'depth_ratio2',
        # Microstructure
        'micro_price', 'slope_ask', 'slope_bid', 'cum_depth_imbalance', 'spread_slope',
        # Momentum & returns
        'log_return', 'momentum_5', 'mid_return',
        # Time encoding
        'sin_time', 'cos_time',
        # Order-flow deltas
        'delta_bid1', 'delta_ask1', 'delta_bid2', 'delta_ask2'
    ]
    cols = feature_cols if feature_cols is not None else default_features

    X_list, y_list, time_ids, start_times = [], [], [], []

    for time_id, group in df.groupby('time_id'):
        group = group.reset_index(drop=True)
        n = len(group)
        for i in range(0, n - window - horizon + 1, step):
            seq = group.iloc[i:i+window]
            X_list.append(seq[cols].values)

            future = group.iloc[i+window:i+window+horizon]
            y_list.append(np.sqrt(np.sum(future['log_return'].values ** 2)))
            time_ids.append(time_id)
            start_times.append(group.iloc[i]['seconds_in_bucket'])

    return pd.DataFrame({
        'X': X_list,
        'y': y_list,
        'time_id': time_ids,
        'start_time': start_times
    })


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
