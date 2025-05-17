import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# Set seed
np.random.seed(3888)

# Stock name map
stock_names = {
    50200: "SPY XNAS",
    104919: "QQQ XNAS",
    22771: "NFLX XNAS"
}

# ===========
# Load model + scalers
# ===========
@st.cache_resource
def load_lstm_model_and_scalers():
    model = load_model("Archive/baseline.h5", compile=False)
    scalers = joblib.load("Archive/baseline_scalers.pkl")
    return model, scalers["x_scaler"], scalers["y_scaler"]

@st.cache_data
def load_preprocessed_data(stock_id):
    return pd.read_pickle(f"data/preprocessed_{stock_id}.pkl")

# ===========
# User selection
# ===========
options = [f"{sid} – {stock_names[sid]}" for sid in stock_names]
selected_label = st.selectbox("Choose a stock:", options)
stock_id = int(selected_label.split(" – ")[0])

# ===========
# Prediction
# ===========
with st.spinner("Loading and predicting..."):
    df = load_preprocessed_data(stock_id)
    model, x_scaler, y_scaler = load_lstm_model_and_scalers()

    X = df.drop(columns=["realized_volatility"])
    y_true = df["realized_volatility"].values

    X_scaled = x_scaler.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    y_pred_scaled = model.predict(X_lstm).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# ===========
# Plot
# ===========
selected_steps = st.slider("Forecast steps to display", 20, min(len(y_true), 200), 100)

fig = go.Figure()
fig.add_trace(go.Scatter(y=y_true[:selected_steps], name="Actual", mode="lines"))
fig.add_trace(go.Scatter(y=y_pred[:selected_steps], name="Predicted", mode="lines"))
fig.update_layout(title="LSTM Forecast vs Actual", xaxis_title="Step", yaxis_title="Volatility", template="plotly_white")
st.plotly_chart(fig)

# ===========
# Metrics
# ===========
def evaluate(true, pred):
    pred = np.clip(pred, 1e-8, None)
    true = np.clip(true, 1e-8, None)
    mse = np.mean((true - pred) ** 2)
    qlike = np.mean(np.log(pred**2) + (true**2) / (pred**2))
    return mse, qlike

mse_val, qlike_val = evaluate(y_true[:selected_steps], y_pred[:selected_steps])

st.subheader("LSTM Evaluation Metrics")
st.write(f"**MSE**: {mse_val:.8f}")
st.write(f"**QLIKE**: {qlike_val:.8f}")
st.caption("Note: Each step ≈ 10 seconds of trading time.")
