# install packages
# pip install shiny shinywidgets faicons seaborn statsmodels pandas numpy scikit-learn matplotlib tensorflow

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
from tensorflow.keras.models import load_model

stock_ids = {
    22771: "22771: NFLX XNAS",
    104919: "104919: QQQ XNAS",
    50200: "50200: SPY XNAS",
}

ui.page_opts(title="Volatility prediction dashboard", fillable=True)

with ui.sidebar(title="Model Selection"):
    ui.input_select("selected_stock_id", "Select a Stock ID you want to predict:", stock_ids, selected=50200)


with ui.layout_column_wrap(fill=False):
    with ui.value_box(showcase=icon_svg("chart-line")):
        "Q-like"

        @render.text
        def qlike():
            actual, predicted = prediction()
            return f"{round(model.qlike(actual, predicted), 3)}"

    with ui.value_box(showcase=icon_svg("xmark")):
        "MSE"

        @render.text
        def mse():
            actual, predicted = prediction()
            return f"{round(model.mse(actual, predicted), 3)}"

    with ui.value_box(showcase=icon_svg("ruler-combined")):
        "RMSE"

        @render.text
        def rmse():
            actual, predicted = prediction()
            return f"{round(model.rmse(actual, predicted), 3)}"
        
    with ui.value_box(showcase=icon_svg("clock")):
        "Computation Time (s)"

        @render.text
        def comp_time():
            prediction()  # trigger calculation
            return f"{round(prediction.elapsed, 3)}"



with ui.layout_columns():
    with ui.card(full_screen=True):
        ui.card_header("Volatility Forecasts")

        @render.plot
        def predict():
            actual, predicted = prediction()

            ax = sns.lineplot(x=range(len(actual)), y=actual, label="Actual", linestyle="-", color="blue")
            sns.lineplot(x=range(len(predicted)), y=predicted, label="Predicted", linestyle="--", color="orange", ax=ax)

            ax.set_title("Actual vs Predicted Volatility")
            ax.set_xlabel("Time")
            ax.set_ylabel("Realized Volatility")

            return ax.figure


# ui.include_css("styles.css")

# Reactive prediction logic with timing
@reactive.calc
def prediction():
    start_time = time.time()

    model_path = "models/out/lstm/config_v256d03_more_feature.h5"
    scaler_path = "models/out/lstm/config_v256d03_more_feature_scalers.pkl"

    selected_model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    load_data = model.load_data(input.selected_stock_id())
    X = load_data[["rv_lag_1", "rv_lag_5", "rv_lag_10"]]
    y = load_data["realized_volatility"]

    X_scaled = scaler.transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    actual = y.values
    predicted = selected_model.predict(X_lstm).flatten()

    end_time = time.time()
    prediction.elapsed = end_time - start_time

    return actual, predicted
