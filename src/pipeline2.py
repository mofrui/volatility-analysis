import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor


random.seed(3888)
np.random.seed(3888)

def train_bid_ask_spread_model(
    df_main: pd.DataFrame,
    pred_df: pd.DataFrame,
    window_size: int = 330,
    step: int = 10,
    train_frac: float = 0.8,
    cache_dir: str = "models/pipeline2",
    model_save_path: str = "models/pipeline2/bid_ask_spread_model.pkl",
    grid_params: dict = None
):
    """
    Trains an XGBoost model to predict bid-ask spread using rolling features
    and LSTM volatility predictions, with all seeds fixed to 3888.
    """
    random.seed(3888)
    np.random.seed(3888)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    rolling_rows = []
    for time_id, grp in df_main.sort_values(['time_id','seconds_in_bucket']).groupby('time_id'):
        grp = grp.reset_index(drop=True)
        for i in range(0, len(grp) - window_size + 1, step):
            window = grp.iloc[i:i+window_size]
            rolling_rows.append({
                'time_id': time_id,
                'start_time': window.iloc[0]['seconds_in_bucket'],
                'wap': window['wap'].mean(),
                'spread_pct': window['spread_pct'].mean(),
                'imbalance': window['imbalance'].mean(),
                'depth_ratio': window['depth_ratio'].mean(),
                'log_return': window['log_return'].mean(),
                'bid_ask_spread': window['spread'].mean()
            })
    rolling_df = pd.DataFrame(rolling_rows)
    
    rolling_df['bid_ask_spread_lead1'] = (
        rolling_df.groupby('time_id')['bid_ask_spread']
                  .shift(-1)
    )
    rolling_df = rolling_df.dropna().reset_index(drop=True)
    merged = pd.merge(
        pred_df.rename(columns={'y_pred':'predicted_volatility_lead1'}),
        rolling_df,
        on=['time_id','start_time'],
        how='inner'
    )
    
    y = merged['bid_ask_spread_lead1']
    X = merged.drop(columns=['time_id','start_time','bid_ask_spread_lead1'])
    
    cutoff = int(len(merged) * train_frac)
    X_train, X_test = X.iloc[:cutoff].copy(), X.iloc[cutoff:].copy()
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
    
    scaler = StandardScaler()
    for col in ['spread_pct']:
        X_train[f"{col}_scaled"] = scaler.fit_transform(X_train[[col]])
        X_test[f"{col}_scaled"]  = scaler.transform(X_test[[col]])
    
    feature_cols = [
        'predicted_volatility_lead1', 'spread_pct_scaled',
        'wap', 'imbalance', 'depth_ratio', 'log_return', 'bid_ask_spread'
    ]
    X_train_final = X_train[feature_cols]
    X_test_final  = X_test[feature_cols]
    
    if grid_params is None:
        grid_params = {
            'n_estimators': [100, 300],
            'max_depth': [5, 6],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 1],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 5]
        }
    
    xgb = XGBRegressor(random_state=3888, objective='reg:squarederror')
    grid = GridSearchCV(
        xgb, grid_params, cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=0
    )
    grid.fit(X_train_final, y_train)
    best_model = grid.best_estimator_
    
    y_pred = best_model.predict(X_test_final)
    metrics = {
        'MSE':  mean_squared_error(y_test, y_pred),
        'MAE':  mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2':   r2_score(y_test, y_pred)
    }
    
    joblib.dump(best_model, model_save_path)
    return best_model, metrics


import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def generate_quote(
    pred_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    spread_model_path: str,
    stock_id: int,
    window_size: int = 330,
    step: int = 10,
    random_state: int = 42
) -> dict:
    """
    Generate bid and ask quotes for a given stock using LSTM volatility predictions and XGBoost spread model.
    
    Parameters
    ----------
    pred_csv : str
        Path to CSV with LSTM predictions (must have ['time_id','start_time','y_pred']).
    snapshot_df : pd.DataFrame
        Original snapshot DataFrame including order book columns:
        ['stock_id','time_id','seconds_in_bucket','wap','spread_pct',
         'imbalance','depth_ratio','log_return','bid_price1','ask_price1','bid_ask_spread'].
    spread_model_path : str
        Path to the trained XGBoost model (.pkl file).
    stock_id : int
        The stock ID to generate quotes for.
    window_size : int
        Number of rows per rolling window (default 330).
    step : int
        Step size between windows (default 10).
    random_state : int
        Seed for reproducible sampling (default 42).
    
    Returns
    -------
    dict
        Dictionary with keys: 'pred_mid_price', 'pred_spread', 'bid', 'ask', 
        and optionally 'actual_mid', 'actual_spread', 'real_bid', 'real_ask' if available.
    """
    # Load predictions
    pred_df.rename(columns={"y_pred": "predicted_volatility_lead1"})
    
    # Prepare snapshot for specified stock
    df = snapshot_df.copy()
    df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2
    df = (
        df[df['stock_id'] == stock_id]
          .sort_values(['time_id','seconds_in_bucket'])
          .reset_index(drop=True)
    )
    
    # Compute rolling windows
    rolling_rows = []
    for time_id, group in df.groupby("time_id"):
        group = group.reset_index(drop=True)
        for i in range(0, len(group) - window_size + 1, step):
            win = group.iloc[i:i+window_size]
            rolling_rows.append({
                'time_id': time_id,
                'start_time': win.iloc[0]['seconds_in_bucket'],
                'mid_price': win['mid_price'].mean(),
                'wap': win['wap'].mean(),
                'spread_pct': win['spread_pct'].mean(),
                'imbalance': win['imbalance'].mean(),
                'depth_ratio': win['depth_ratio'].mean(),
                'log_return': win['log_return'].mean(),
                'bid_ask_spread': win['spread'].mean()
            })
    rolling_df = pd.DataFrame(rolling_rows)
    
    # Lead target and merge
    rolling_df['bid_ask_spread_lead1'] = (
        rolling_df.groupby('time_id')['bid_ask_spread'].shift(-1)
    )
    rolling_df = rolling_df.dropna().reset_index(drop=True)
    merged = pd.merge(
        pred_df,
        rolling_df,
        on=['time_id','start_time'],
        how='inner'
    )
    
    # Sample a row for quoting
    row = merged.sample(1, random_state=random_state).iloc[0]
    next_idx = merged.index.get_loc(row.name) + 1
    
    # Predicted mid price
    pred_mid = row['mid_price']
    
    # Scale spread_pct
    scaler = StandardScaler()
    scaled_spread = scaler.fit_transform([[row['spread_pct']]])[0,0]
    
    # Load spread model and predict spread
    spread_model = joblib.load(spread_model_path)
    feat = np.array([
        row['predicted_volatility_lead1'],
        scaled_spread,
        row['wap'],
        row['imbalance'],
        row['depth_ratio'],
        row['log_return'],
        row['bid_ask_spread']
    ]).reshape(1, -1)
    pred_spread = spread_model.predict(feat)[0]
    
    # Compute quotes
    bid = pred_mid - pred_spread / 2
    ask = pred_mid + pred_spread / 2
    
    # Prepare result
    result = {
        'pred_mid_price': pred_mid,
        'pred_spread': pred_spread,
        'bid': bid,
        'ask': ask
    }
    
    # Optionally include actual next window values
    if next_idx < len(merged):
        actual = merged.iloc[next_idx]
        actual_mid = actual['mid_price']
        actual_spread = actual['bid_ask_spread']
        result.update({
            'actual_mid': actual_mid,
            'actual_spread': actual_spread,
            'real_bid':  actual_mid - actual_spread/2,
            'real_ask':  actual_mid + actual_spread/2
        })
    
    return result


def evaluate_quote_strategy(
    pred_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    spread_model_path: str,
    window_size: int = 330,
    step: int = 10
) -> dict:
    """
    Evaluate the bid-ask quoting strategy using LSTM volatility predictions
    and the trained XGBoost spread model. Prints summary metrics and
    visualizes quote effectiveness.
    
    Parameters
    ----------
    pred_csv : str
        Path to CSV with LSTM predictions including ['time_id','start_time','y_pred'].
    snapshot_df : pd.DataFrame
        Preprocessed snapshot DataFrame with columns:
        ['stock_id','time_id','seconds_in_bucket','wap','spread_pct',
         'imbalance','depth_ratio','log_return','bid_price1','ask_price1','bid_ask_spread'].
    spread_model_path : str
        Path to the trained XGBoost model (.pkl).
    window_size : int, optional
        Rolling window size in seconds (default=330).
    step : int, optional
        Step between windows in seconds (default=10).
        
    Returns
    -------
    metrics : dict
        Evaluation metrics: hit_ratio, avg_effectiveness, inside_spread_ratio, sharpe.
    """
    # Load model and predictions
    model = joblib.load(spread_model_path)
    pred_df.rename(columns={"y_pred": "predicted_volatility_lead1"})
    
    # Preprocess snapshot
    df = snapshot_df.copy()
    df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2
    df = df.sort_values(['time_id', 'seconds_in_bucket']).reset_index(drop=True)
    
    # Build rolling windows
    rolling = []
    for time_id, grp in df.groupby("time_id"):
        grp = grp.reset_index(drop=True)
        for i in range(0, len(grp) - window_size + 1, step):
            win = grp.iloc[i:i+window_size]
            rolling.append({
                'time_id': time_id,
                'start_time': win.iloc[0]['seconds_in_bucket'],
                'wap': win['wap'].mean(),
                'spread_pct': win['spread_pct'].mean(),
                'imbalance': win['imbalance'].mean(),
                'depth_ratio': win['depth_ratio'].mean(),
                'log_return': win['log_return'].mean(),
                'bid_ask_spread': win['spread'].mean()
            })
    rolling_df = pd.DataFrame(rolling)
    rolling_df['bid_ask_spread_lead1'] = (
        rolling_df.groupby('time_id')['bid_ask_spread'].shift(-1)
    )
    rolling_df.dropna(inplace=True)
    
    # Merge with LSTM predictions
    merged = pd.merge(pred_df, rolling_df, on=['time_id','start_time'], how='inner')
    
    # Scale spread_pct
    scaler = StandardScaler()
    merged['spread_pct_scaled'] = scaler.fit_transform(merged[['spread_pct']])
    
    # Predict spread
    features = [
        'predicted_volatility_lead1',
        'spread_pct_scaled',
        'wap','imbalance','depth_ratio','log_return','bid_ask_spread'
    ]
    X = merged[features]
    merged['predicted_bid_ask_spread'] = model.predict(X)
    
    # Construct quotes
    merged['pred_mid_price'] = merged['wap']
    merged['quoted_bid'] = merged['pred_mid_price'] - merged['predicted_bid_ask_spread']/2
    merged['quoted_ask'] = merged['pred_mid_price'] + merged['predicted_bid_ask_spread']/2
    
    # Real next-window quotes
    merged['real_bid'] = merged['wap'] - merged['bid_ask_spread_lead1']/2
    merged['real_ask'] = merged['wap'] + merged['bid_ask_spread_lead1']/2
    
    # Evaluation flags
    merged['both_hit'] = (merged['quoted_bid'] >= merged['real_bid']) & \
                        (merged['quoted_ask'] <= merged['real_ask'])
    merged['quote_effectiveness'] = (
        (merged['quoted_bid'] - merged['real_bid']) +
        (merged['real_ask'] - merged['quoted_ask'])
    ) / 2
    merged['inside_spread'] = (merged['quoted_bid'] > merged['real_bid']) & \
                              (merged['quoted_ask'] < merged['real_ask'])
    
    # Summary metrics
    hit_ratio = merged['both_hit'].mean()
    avg_eff   = merged['quote_effectiveness'].mean()
    inside_ratio = merged['inside_spread'].mean()
    sharpe    = avg_eff / merged['quote_effectiveness'].std()
    metrics = {
        'hit_ratio': hit_ratio,
        'avg_effectiveness': avg_eff,
        'inside_spread_ratio': inside_ratio,
        'sharpe_ratio': sharpe
    }
    
    # Print summary
    print("Quote Evaluation Metrics:")
    print(f"1. Hit Ratio:                  {hit_ratio:.2%}")
    print(f"2. Avg. Quote Effectiveness:  {avg_eff:.6f}")
    print(f"3. Inside-Spread Ratio:       {inside_ratio:.2%}")
    print(f"4. Sharpe Ratio:              {sharpe:.4f}")
    
    # Visualizations
    plt.figure(figsize=(12, 4))
    plt.plot(merged.index, merged['quote_effectiveness'], label='Quote Effectiveness')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Quote Effectiveness Over Time")
    plt.xlabel("Window Index")
    plt.ylabel("Effectiveness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return metrics
