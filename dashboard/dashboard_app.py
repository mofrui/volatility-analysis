# install packages
# pip install shiny shinywidgets faicons seaborn statsmodels pandas numpy scikit-learn matplotlib plotly tensorflow-macos tensorflow-metal pyarrow

# to run: shiny run --reload dashboard/dashboard_app.py 

from faicons import icon_svg
import seaborn as sns
from shinywidgets import render_plotly
from shiny import reactive, req
from shiny.express import input, render, ui
import model
import statsmodels.api as sm
import time
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


stock_ids = {
    22771: "22771: NFLX XNAS",
    104919: "104919: QQQ XNAS",
    50200: "50200: SPY XNAS",
}

def qlike(y_true, y_pred):
    y_true = np.clip(y_true, 1e-8, None)
    y_pred = np.clip(y_pred, 1e-8, None)
    return np.mean(np.log(y_pred**2) + (y_true**2) / (y_pred**2))

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


ui.page_opts(title="Volatility prediction dashboard", fillable=True)

with ui.sidebar(title="Model Selection"):
    ui.input_select("selected_stock_id", "Select a Stock ID you want to predict:", stock_ids, selected=50200)



# ------------------------------------------
# CACHED RESOURCES (Model + Scaler)
# ------------------------------------------

@reactive.calc
def loaded_model():
    return load_model("models/out/lstm/config_v256d03_more_feature.h5", compile=False)

@reactive.calc
def loaded_scaler():
    return joblib.load("models/out/lstm/config_v256d03_more_feature_scalers.pkl")["x_scaler"]

def load_data(stock_id: int) -> pd.DataFrame:
    import pandas as pd
    from models import util

    folder_path = "data"
    feature_path = os.path.join(folder_path, "order_book_feature.parquet")
    target_path = os.path.join(folder_path, "order_book_target.parquet")

    df_feat = pd.read_parquet(feature_path)
    df_tgt = pd.read_parquet(target_path)

    # Filter for selected stock
    df_feat = df_feat[df_feat["stock_id"] == stock_id].copy()
    df_tgt = df_tgt[df_tgt["stock_id"] == stock_id].copy()

    # Merge feature and target data
    df = pd.merge(df_feat, df_tgt, on=["stock_id", "time_id", "seconds_in_bucket"], how="inner")
    df = df.sort_values(by=['stock_id', 'time_id', 'seconds_in_bucket']).reset_index(drop=True)

    # Step 1: Compute snapshot features
    df = util.create_snapshot_features(df)
    df = util.add_features(df)

    # ✅ Filter time_ids with sufficient length for sequences
    valid_ids = df.groupby("time_id").filter(lambda g: len(g) >= 340)["time_id"].unique()
    df = df[df["time_id"].isin(valid_ids)].copy()

    # Step 2: Generate sequences on filtered time_ids
    feature_cols = [
        "wap", "spread_pct", "imbalance", "depth_ratio", "log_return",
        "log_wap_change", "rolling_std_logret", "spread_zscore", "volume_imbalance"
    ]
    lstm_df = util.generate_tick_sequences(df, feature_cols)

    # If lstm_df is empty, handle gracefully
    if lstm_df.empty:
        return pd.DataFrame(columns=feature_cols + ["realized_volatility"])

    # Step 3: Flatten for dashboard usage
    X = pd.DataFrame(np.stack(lstm_df["X"].values).squeeze())
    X.columns = feature_cols
    X["realized_volatility"] = lstm_df["y"].values

    return X


with ui.layout_column_wrap(fill=False):
    with ui.value_box(showcase=icon_svg("chart-line")):
        "Q-like"

        @render.text
        def qlike():
            actual, predicted = prediction()
            return f"{round(qlike(actual, predicted), 3)}"

    with ui.value_box(showcase=icon_svg("xmark")):
        "MSE"

        @render.text
        def mse():
            actual, predicted = prediction()
            return f"{round(mse(actual, predicted), 3)}"

    with ui.value_box(showcase=icon_svg("ruler-combined")):
        "RMSE"

        @render.text
        def rmse():
            actual, predicted = prediction()
            return f"{round(rmse(actual, predicted), 3)}"
        
    with ui.value_box(showcase=icon_svg("clock")):
        "Computation Time (s)"

        @render.text
        def comp_time():
            prediction()  # trigger calculation
            return f"{round(prediction.elapsed, 3)}"



with ui.layout_columns():
    with ui.card(full_screen=True):
        ui.card_header("Volatility Forecasts")

        @render_plotly
        def line_plot():
            actual, predicted = prediction()
            x_vals = list(range(len(actual)))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=actual, mode="lines", name="Actual", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=x_vals, y=predicted, mode="lines", name="Predicted", line=dict(color="orange", dash="dash")))

            fig.update_layout(
                title="Actual vs Predicted Volatility",
                xaxis_title="Time",
                yaxis_title="Realized Volatility",
                template="plotly_white",
                margin=dict(l=40, r=40, t=40, b=40),
            )

            return fig






# ui.include_css("styles.css")

# Reactive prediction logic with timing
@reactive.calc
def prediction():
    start_time = time.time()

    selected_model = loaded_model()

    x_scaler = loaded_scaler()

    df = load_data(input.selected_stock_id())
    X = df[[
        "wap", "spread_pct", "imbalance", "depth_ratio", "log_return",
        "log_wap_change", "rolling_std_logret", "spread_zscore", "volume_imbalance"
    ]]
    y = df["realized_volatility"]

    X_scaled = x_scaler.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    actual = y.values

    predicted = selected_model.predict(X_lstm).flatten()

    prediction.elapsed = time.time() - start_time
    return actual, predicted
