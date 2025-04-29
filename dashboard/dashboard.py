import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import joblib
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Set random seed
np.random.seed(3888)
random.seed(3888)

# =====================
# 1. Load Data
# =====================
folder_path = "/Users/dais/Downloads/Optiver_additional data"

feature_path = os.path.join(folder_path, "order_book_feature.parquet")
target_path = os.path.join(folder_path, "order_book_target.parquet")

feature_df = pd.read_parquet(feature_path, engine='pyarrow')
target_df = pd.read_parquet(target_path, engine='pyarrow')
combined_df = pd.concat([feature_df, target_df], axis=0).sort_values(['stock_id', 'time_id', 'seconds_in_bucket'])

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

feature_engineered_df = compute_orderbook_features(combined_df)

# Select stock
stock_id = st.selectbox("Choose a stock_id:", feature_engineered_df['stock_id'].unique())
stock_df = feature_engineered_df[feature_engineered_df["stock_id"] == stock_id]

# Compute realized volatility
stock_df["log_return"] = stock_df.groupby("time_id")["wap"].transform(lambda x: np.log(x / x.shift(1)))
rv_df = stock_df.groupby("time_id")["log_return"].agg(lambda x: np.sqrt(np.sum(x**2))).reset_index()
rv_df = rv_df.rename(columns={"log_return": "realized_volatility"})

# Create lag features
hav_df = rv_df.copy()
hav_df["rv_lag_1"] = hav_df["realized_volatility"].shift(1)
hav_df["rv_lag_5"] = hav_df["realized_volatility"].shift(5)
hav_df["rv_lag_10"] = hav_df["realized_volatility"].shift(10)
hav_df = hav_df.dropna().reset_index(drop=True)

X = hav_df[["rv_lag_1", "rv_lag_5", "rv_lag_10"]]
y = hav_df["realized_volatility"]

# =====================
# 2. Load Models
# =====================

@st.cache_resource
def load_models():
    ols = joblib.load("research/models/ols_model.pkl")
    wls = joblib.load("research/models/wls_model.pkl")
    ridge = joblib.load("research/models/ridge_model.pkl")
    ridge_scaler = joblib.load("research/models/ridge_scaler.pkl")
    ridgecv = joblib.load("research/models/ridgecv_model.pkl")
    ridgecv_scaler = joblib.load("research/models/ridgecv_scaler.pkl")
    lassocv = joblib.load("research/models/lassocv_model.pkl")
    lassocv_scaler = joblib.load("research/models/lasso_cv_scaler.pkl")
    return ols, wls, ridge, ridge_scaler, ridgecv, ridgecv_scaler, lassocv, lassocv_scaler

ols_model, wls_model, ridge_fixed_model, ridge_fixed_scaler, ridge_cv_model, ridge_cv_scaler, lasso_cv_model, lasso_cv_scaler = load_models()

# =====================
# 3. Predict using models
# =====================
X_const = sm.add_constant(X)

preds = {
    "Actual": y.values,
    "OLS": ols_model.predict(X_const),
    "WLS": wls_model.predict(X_const),
    "Ridge (fixed alpha=1)": ridge_fixed_model.predict(ridge_fixed_scaler.transform(X)),
    "RidgeCV (best alpha)": ridge_cv_model.predict(ridge_cv_scaler.transform(X)),
    "LassoCV (best alpha)": lasso_cv_model.predict(lasso_cv_scaler.transform(X)),
}

# Evaluate
def evaluate(true, pred):
    mse = np.mean((true - pred)**2)
    qlike = np.mean(np.log(np.clip(pred, 1e-8, None)**2) + (np.clip(true, 1e-8, None)**2) / (np.clip(pred, 1e-8, None)**2))
    return mse, qlike

metrics = {}
for model_name, pred in preds.items():
    if model_name != "Actual":
        mse, qlike = evaluate(y, pred)
        metrics[model_name] = {"MSE": mse, "QLIKE": qlike}

# =====================
# 4. Streamlit Interface
# =====================
selected_models = st.multiselect(
    "Select models to display:",
    list(preds.keys()),
    default=["Actual", "OLS"]
)

fig = go.Figure()

for model in selected_models:
    fig.add_trace(go.Scatter(y=preds[model], mode='lines', name=model))

fig.update_layout(
    title="Volatility Forecasts",
    xaxis_title="Forecast Step",
    yaxis_title="Volatility",
    template="plotly_white"
)

st.plotly_chart(fig)

# Check if we want to use saved metrics
if stock_id == 50200:
    # Load pre-saved metrics
    metrics_df = pd.read_json("research/models/50200_metrics.json")
    metrics_dict = metrics_df.set_index("Model").T.to_dict()
    use_saved_metrics = True
elif stock_id == 104919:
    # No saved metrics for 104919 yet — recompute
    use_saved_metrics = False
else:
    # Other stocks — recompute
    use_saved_metrics = False

# Show metrics table
table_data = []
for model_name in selected_models:
    if model_name != "Actual":
        if use_saved_metrics:
            mse = metrics_dict[model_name]["MSE"]
            qlike = metrics_dict[model_name]["QLIKE"]
        else:
            mse = metrics[model_name]["MSE"]
            qlike = metrics[model_name]["QLIKE"]
        
        table_data.append({
            "Model": model_name,
            "MSE": f"{mse:.8f}",
            "QLIKE": f"{qlike:.8f}"
        })

if table_data:
    st.dataframe(pd.DataFrame(table_data))
