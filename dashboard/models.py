import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os


def load_parquet_files(folder_path):
    feature_path = os.path.join(folder_path, "order_book_feature.parquet")
    target_path = os.path.join(folder_path, "order_book_target.parquet")

    feature_df = pd.read_parquet(feature_path, engine="pyarrow")
    target_df = pd.read_parquet(target_path, engine="pyarrow")

    combined_df = pd.concat([feature_df, target_df], axis=0)
    combined_df = combined_df.sort_values(by=["stock_id", "time_id", "seconds_in_bucket"]).reset_index(drop=True)
    
    return combined_df

def load_time_reference(folder_path):
    time_ref_path = os.path.join(folder_path, "time_id_reference.csv")
    time_ref_df = pd.read_csv(time_ref_path)
    time_ref_df["datetime"] = pd.to_datetime(time_ref_df["date"] + " " + time_ref_df["time"])
    return time_ref_df

def merge_datetime(df, time_ref_df):
    df = df.drop(columns=["datetime"], errors="ignore")
    df = pd.merge(df, time_ref_df[["time_id", "datetime"]], on="time_id", how="left")
    return df

def filter_stock(df, stock_id):
    return df[df["stock_id"] == stock_id].copy()

def split_train_test(df, split_ratio=0.8):
    df = df.sort_values(by="time_id")
    unique_ids = sorted(df["time_id"].unique())
    cutoff = int(len(unique_ids) * split_ratio)
    train_ids = unique_ids[:cutoff]
    test_ids = unique_ids[cutoff:]
    train_df = df[df["time_id"].isin(train_ids)].copy()
    test_df = df[df["time_id"].isin(test_ids)].copy()
    return train_df, test_df

def add_lag_features(rv_df):
    rv_df["rv_lag_1"] = rv_df["realized_volatility"].shift(1)
    rv_df["rv_lag_5"] = rv_df["realized_volatility"].shift(5)
    rv_df["rv_lag_10"] = rv_df["realized_volatility"].shift(10)
    rv_df = rv_df.dropna().reset_index(drop=True)
    return rv_df


# ============================
# Feature Engineering
# ============================
def compute_orderbook_features(df):
    df = df.copy()
    df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2
    df['wap'] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['ask_size1'] + df['bid_size1'])
    df['bid_ask_spread'] = df['ask_price1'] - df['bid_price1']
    df['spread_pct'] = df['bid_ask_spread'] / df['mid_price']
    df['spread_variation'] = df.groupby(['stock_id', 'time_id'])['spread_pct'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
    df['imbalance'] = (df['bid_size1'] - df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    df['depth_ratio'] = df['bid_size1'] / df['ask_size1'].replace(0, np.nan)
    return df

def compute_realized_volatility(df):
    df["log_return"] = df.groupby("time_id")["wap"].transform(lambda x: np.log(x / x.shift(1)))
    rv_df = df.groupby("time_id")["log_return"].agg(lambda x: np.sqrt(np.sum(x**2))).reset_index()
    rv_df = rv_df.rename(columns={"log_return": "realized_volatility"})
    return rv_df

# ============================
# Modeling
# ============================
def rolling_window_forecast(hav_df, model_type="ols", W=330, H=10, S=5, alpha=None, alphas=None):
    all_preds, all_actuals = [], []
    best_alpha = None

    for start in range(0, len(hav_df) - W - H + 1, S):
        train_window = hav_df.iloc[start:start + W]
        test_window = hav_df.iloc[start + W:start + W + H]

        X_train = train_window[['rv_lag_1', 'rv_lag_5', 'rv_lag_10']]
        y_train = train_window['realized_volatility']
        X_test = test_window[['rv_lag_1', 'rv_lag_5', 'rv_lag_10']]

        if model_type in ["ridge", "lasso", "ridge_cv", "lasso_cv"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if model_type == "ols":
            X_train = sm.add_constant(X_train)
            X_test = sm.add_constant(X_test)
            model = sm.OLS(y_train, X_train).fit()
            preds = model.predict(X_test)

        elif model_type == "wls":
            weights = 1 / (y_train ** 2 + 1e-8)
            X_train = sm.add_constant(X_train)
            X_test = sm.add_constant(X_test)
            model = sm.WLS(y_train, X_train, weights=weights).fit()
            preds = model.predict(X_test)

        elif model_type == "ridge":
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        elif model_type == "lasso":
            model = LassoCV(alphas=[alpha], cv=5, max_iter=5000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        elif model_type == "ridge_cv":
            model = RidgeCV(alphas=alphas, cv=5)
            model.fit(X_train, y_train)
            if best_alpha is None:
                best_alpha = model.alpha_
            preds = model.predict(X_test)

        elif model_type == "lasso_cv":
            model = LassoCV(alphas=alphas, cv=5, max_iter=5000)
            model.fit(X_train, y_train)
            if best_alpha is None:
                best_alpha = model.alpha_
            preds = model.predict(X_test)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        all_preds.extend(preds)
        all_actuals.extend(test_window['realized_volatility'].values)

    if model_type in ["ridge_cv", "lasso_cv"]:
        return np.array(all_actuals), np.array(all_preds), best_alpha
    else:
        return np.array(all_actuals), np.array(all_preds)

# ============================
# Evaluation
# ============================
def evaluate_model(true, pred):
    pred_clipped = np.clip(pred, 1e-4, None)
    true_clipped = np.clip(true, 1e-4, None)
    mse = mean_squared_error(true_clipped, pred_clipped)
    qlike_score = np.mean(np.log(pred_clipped**2) + (true_clipped**2) / (pred_clipped**2))
    return mse, qlike_score
