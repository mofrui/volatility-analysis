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

stock_names = {
    50200: "SPY XNAS",
    104919: "QQQ XNAS",
    22771: "NFLX XNAS"
}

# Paths
folder_path = "/Users/dais/Downloads/Optiver_additional data"
feature_path = os.path.join(folder_path, "order_book_feature.parquet")
target_path = os.path.join(folder_path, "order_book_target.parquet")

# Load feature/target metadata (only to list available stocks)
feature_df = pd.read_parquet(feature_path, engine='pyarrow')
target_df = pd.read_parquet(target_path, engine='pyarrow')

# include the stock we trained on and the most correlated and least correlated stock with the stock we used for training 
stock_ids = [50200, 104919,22771]

# Stock selector
# Use the name alongside the ID in the selectbox
options = [f"{sid} – {stock_names[sid]}" for sid in stock_ids]
selected_label = st.selectbox("Choose a stock:", options)

# Extract the numeric stock_id for internal use
stock_id = int(selected_label.split(" – ")[0])


# =====================
# 1. Functions
# =====================
def compute_orderbook_features(df):
    df = df.copy()
    df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2
    df['wap'] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['ask_size1'] + df['bid_size1'])
    df['bid_ask_spread'] = df['ask_price1'] - df['bid_price1']
    df['spread_pct'] = df['bid_ask_spread'] / df['mid_price']
    df['spread_variation'] = df.groupby('time_id')['spread_pct'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
    df['imbalance'] = (df['bid_size1'] - df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    df['depth_ratio'] = df['bid_size1'] / df['ask_size1'].replace(0, np.nan)
    return df

def evaluate(true, pred):
    pred = np.clip(pred, 1e-8, None)
    true = np.clip(true, 1e-8, None)
    mse = np.mean((true - pred)**2)
    qlike = np.mean(np.log(pred**2) + (true**2) / (pred**2))
    return mse, qlike

@st.cache_resource
def load_models():
    ols = joblib.load("research/models/ols_model.pkl")
    wls = joblib.load("research/models/wls_model.pkl")
    return ols, wls

# =====================
# 2. Main Workflow
# =====================
with st.spinner("Loading ..."):
    # Load per-stock data
    stock_feature = feature_df[feature_df["stock_id"] == stock_id].copy()
    stock_target = target_df[target_df["stock_id"] == stock_id].copy()
    stock_df = pd.concat([stock_feature, stock_target], axis=0).sort_values(['time_id', 'seconds_in_bucket'])
    stock_df = compute_orderbook_features(stock_df)

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


    # Estimate average step duration in minutes
    avg_steps = hav_df.groupby("time_id").size().mean()
    step_duration_sec = 3600 / avg_steps
    step_duration_min = step_duration_sec / 60
    st.write(f"⏱ Estimated forecast step size: {step_duration_sec:.2f} seconds (~{step_duration_min:.2f} minutes per step)")



    X = hav_df[["rv_lag_1", "rv_lag_5", "rv_lag_10"]]
    y = hav_df["realized_volatility"]
    X_const = sm.add_constant(X)

    # Load models
    ols_model, wls_model = load_models()

    # Predict
    preds = {
        "Actual": y.values,
        "OLS": ols_model.predict(X_const),
        "WLS": wls_model.predict(X_const),
    }

    # Metrics
    metrics = {}
    for model_name, pred in preds.items():
        if model_name != "Actual":
            mse, qlike = evaluate(y, pred)
            metrics[model_name] = {"MSE": mse, "QLIKE": qlike}
# =====================
# 3. Streamlit Interface
# =====================
selected_models = st.multiselect(
    "Select models to display:",
    list(preds.keys()),
    default=["Actual", "OLS"]
)

# Forecast horizon (define after y exists)
forecast_horizons = {
    "Next 20 seconds": 2,
    "Next 30 seconds": 3,
    "Next 1 minute": 6,
    "Next 2 minutes": 12,
    "Next 5 minutes": 30,
    "Next 10 minutes": 60,
    "Full Horizon": len(y)
}

selected_horizon_label = st.selectbox(
    "Select forecast horizon:", 
    list(forecast_horizons.keys()), 
    index=list(forecast_horizons.keys()).index("Full Horizon")
)
selected_steps = forecast_horizons[selected_horizon_label]

# Plot truncated forecasts
fig = go.Figure()
for model in selected_models:
    fig.add_trace(go.Scatter(
        y=preds[model][:selected_steps],
        mode='lines',
        name=model
    ))
fig.update_layout(
    title=f"Volatility Forecasts ({selected_horizon_label})",
    xaxis_title="Forecast Step",
    yaxis_title="Volatility",
    template="plotly_white"
)
st.plotly_chart(fig)

# Show metrics table based on selected forecast window
table_data = []
for model_name in selected_models:
    if model_name != "Actual":
        mse, qlike = evaluate(y[:selected_steps], preds[model_name][:selected_steps])
        table_data.append({
            "Model": model_name,
            "MSE": f"{mse:.8f}",
            "QLIKE": f"{qlike:.8f}"
        })

if table_data:
    st.subheader("Model Evaluation Metrics")
    st.dataframe(pd.DataFrame(table_data))

# Optional note
st.caption("Note: Each forecast step corresponds to approximately 10 seconds of trading time.")


