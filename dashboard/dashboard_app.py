import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import joblib
from pathlib import Path
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# Set random seed
np.random.seed(3888)
random.seed(3888)

# === Stock settings ===
stock_names = {
    50200: "SPY XNAS",
    104919: "QQQ XNAS",
    22771: "NFLX XNAS"
}
stock_ids = list(stock_names.keys())

# === Load model + scalers ===
@st.cache_resource
def load_lstm_model_and_scalers():
    model = load_model("models/out/lstm/config_v256d03_more_feature.h5", compile=False)
    scalers = joblib.load("models/out/lstm/config_v256d03_more_feature_scalers.pkl")
    return model, scalers["x_scaler"], scalers["y_scaler"]

# === Load preprocessed feature data ===
def load_cached_features(stock_id: int) -> pd.DataFrame:
    pkl_path = f"data/preprocessed_{stock_id}.pkl"
    if Path(pkl_path).exists():
        return pd.read_pickle(pkl_path)
    else:
        st.error(f"No preprocessed data found for stock {stock_id}")
        return pd.DataFrame()

# === Evaluation metric ===
def evaluate(true, pred):
    pred = np.clip(pred, 1e-8, None)
    true = np.clip(true, 1e-8, None)
    mse = np.mean((true - pred)**2)
    qlike = np.mean(np.log(pred**2) + (true**2) / (pred**2))
    return mse, qlike

# === UI: stock + horizon selection ===
options = [f"{sid} – {stock_names[sid]}" for sid in stock_ids]
selected_label = st.selectbox("Choose a stock:", options)
stock_id = int(selected_label.split(" – ")[0])

forecast_horizons = {
    "Next 20 seconds": 2,
    "Next 30 seconds": 3,
    "Next 1 minute": 6,
    "Next 2 minutes": 12,
    "Next 5 minutes": 30,
    "Next 10 minutes": 60,
    "Full Horizon": None
}
selected_horizon_label = st.selectbox("Select forecast horizon:", list(forecast_horizons.keys()))
selected_steps = forecast_horizons[selected_horizon_label]

# === Main logic ===
with st.spinner("Loading and predicting..."):
    model, x_scaler, y_scaler = load_lstm_model_and_scalers()
    df = load_cached_features(stock_id)

    if not df.empty:
        X = df.drop(columns=["realized_volatility"])
        y_true = df["realized_volatility"].values

        # Reshape for LSTM
        X_scaled = x_scaler.transform(X)
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        y_pred_scaled = model.predict(X_lstm).flatten()
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Truncate horizon
        steps = selected_steps or len(y_true)
        y_true_plot = y_true[:steps]
        y_pred_plot = y_pred[:steps]

        # === Plot ===
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_true_plot, mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(y=y_pred_plot, mode="lines", name="Predicted"))
        fig.update_layout(
            title="LSTM Forecast",
            xaxis_title="Forecast Step",
            yaxis_title="Volatility",
            template="plotly_white"
        )
        st.plotly_chart(fig)

        # === Metrics ===
        mse, qlike = evaluate(y_true_plot, y_pred_plot)
        st.subheader("Model Evaluation Metrics")
        st.table(pd.DataFrame([{
            "Model": "LSTM",
            "MSE": f"{mse:.6f}",
            "QLIKE": f"{qlike:.6f}"
        }]))

        st.caption("Note: Each step ~10 seconds of trading time")
