from faicons import icon_svg
import seaborn as sns
from shinywidgets import render_plotly

from shiny import reactive, req
from shiny.express import input, render, ui
import model
import statsmodels.api as sm

stock_ids = {
    22771: "22771: NFLX XNAS",
    104919: "104919: QQQ XNAS",
    50200: "50200: SPY XNAS",
}

ui.page_opts(title="Volatility prediction dashboard", fillable=True)

with ui.sidebar(title="Model Selection"):
    ui.input_file("model", "Select a model", accept=".pkl")
    ui.input_select("selected_stock_id", "Choose a Stock ID you want to predict:", stock_ids, selected=9323)


with ui.layout_column_wrap(fill=False):
    with ui.value_box(showcase=icon_svg("chart-line")):
        "Q-like"

        @render.text
        def qlike():
            actual, predicted = prediction()
            return model.qlike(actual, predicted)

    with ui.value_box(showcase=icon_svg("xmark")):
        "MSE"

        @render.text
        def mse():
            actual, predicted = prediction()
            return model.mse(actual, predicted)

    with ui.value_box(showcase=icon_svg("ruler-combined")):
        "RMSE"

        @render.text
        def rmse():
            actual, predicted = prediction()
            return model.rmse(actual, predicted)


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

@reactive.calc
def prediction():
    fileinfo = input.model()
    req(fileinfo)
    model_name = fileinfo[0]["name"]
    model_path = f"Models/{model_name}"
    selected_model = model.load_model(model_path)
    load_data = model.load_data(input.selected_stock_id())

    X = load_data[["rv_lag_1", "rv_lag_5", "rv_lag_10"]]
    y = load_data["realized_volatility"]
    X_const = sm.add_constant(X)

    actual = y.values
    prediction = selected_model.predict(X_const)

    return actual, prediction