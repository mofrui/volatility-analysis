from shiny import App, Inputs, reactive, render, ui
from faicons import icon_svg
import seaborn as sns
import statsmodels.api as sm
import model
from model import load_model, mse, qlike_loss, rmse
import pickle
import pandas as pd
import os

# --- Sidebar: Stock selection ---
# Volatility Forecast inputs
forecast_sidebar = ui.sidebar(
    ui.input_select(
        "selected_stock_id_forecast", 
        "Choose a Stock ID you want to predict:",
        {
            # 22771: "22771: NFLX XNAS",
            # 104919: "104919: QQQ XNAS",
            50200: "50200: SPY XNAS",
        },
        selected=50200,
    ),
    ui.input_checkbox_group(
        "stock_ids_forecast", 
        "Select Stock ID you want to compare",
        {
            22771: "22771: NFLX XNAS",
            104919: "104919: QQQ XNAS",
            50200: "50200: SPY XNAS"
        }
    ),

    ui.input_slider(
        "min_seconds",
        "Start displaying from (seconds):",
        min=0,
        max=3300,
        value=0,
        step=10
    )

)

# Quoting Strategy inputs (could reuse same widgets, just new IDs)
strategy_sidebar = ui.sidebar(
    ui.input_select(
        "selected_stock_id_strategy",
        "Choose a Stock ID you want to predict:",
        {
            22771: "22771: NFLX XNAS",
            104919: "104919: QQQ XNAS",
            50200: "50200: SPY XNAS",
        },
        selected=22771,
    ),
    ui.input_select(
        "selected_time_id_forecast", 
        "Choose a Time ID:",
        {
            1037: "1037",
            528: "528"
        },
        selected=528,
    ),
    ui.input_checkbox_group(
        "stock_ids_strategy",
        "Select Stock ID you want to compare",
        {
            22771: "22771: NFLX XNAS",
            104919: "104919: QQQ XNAS",
            50200: "50200: SPY XNAS"
        }
    )
)

# --- Define App UI ---
app_ui = ui.page_navbar(
    ui.nav_panel(
    "📘 About This App",
    ui.card(
        ui.card_header("What This Dashboard Does"),
        ui.markdown("""
       This interactive dashboard visualizes and predicts **short-term market volatility**
        using high-frequency limit order book data.

        The **“Volatility Forecast”** tab presents predictions generated using our final model: **LSTM**, 
        along with evaluation metrics and actual vs predicted visualizations.

        The **“Quoting Strategies”** tab gives quoting strategy suggestions based on predicted volatility 
        and its relationship to bid-ask spreads. 

        **Higher predicted volatility typically calls for wider spreads** to manage risk, 
        while lower volatility allows for tighter spreads to stay competitive.


        **What You Can Do:**
        - Select a stock to view predicted vs actual volatility.
        - Compare predicted volatility across multiple stocks.
        - Use the prediction output to guide quoting strategies and market decisions.

        **Why Volatility Matters:**
        - Volatility reflects the speed and magnitude of price movements.
        - Higher volatility typically leads to wider bid-ask spreads as market makers adjust for risk.
        - Predicting short-term volatility helps optimize quote placement, balancing risk and profitability.

        **About the Selected Stocks:**
        The dashboard includes three representative stocks to evaluate model performance across varying correlation structures:
        1. **Stock 50200 (SPY XNAS)** – used to train the LSTM model. It tracks a broad market ETF and serves as the primary benchmark.
        2. **Stock 104919 (QQQ XNAS)** – selected as the **most correlated** stock based on mean log return analysis. It represents a tech-heavy index with strong alignment to SPY XNAS.
        3. **Stock 22753 (NFLX XNAS)** – chosen as the **least correlated** stock based on mean log return analysis.
        """)
    ),
    {"class": "bslib-page-dashboard"},
),


    ui.nav_panel(
        "📊 Volatility Forecast",  # Combined tab name
        ui.layout_sidebar(
            forecast_sidebar,
            ui.layout_columns(
                ui.value_box(title="Q-like", value=ui.output_text("qlike")),
                ui.value_box(title="MSE", value=ui.output_text("mse")),
                ui.value_box(title="RMSE", value=ui.output_text("rmse")),
            ),
            ui.card(
                ui.card_header("Actual vs Predicted Volatility"),
                ui.output_plot("predict_plot")
            )
        ),
        {"class": "bslib-page-dashboard"},
    ),
    ui.nav_panel(
        "📊 Quoting Strategies",
         ui.layout_sidebar(
            strategy_sidebar,  
            ui.card(
                ui.card_header("Volatility-Informed Quoting Strategy"),
                ui.output_plot("display_prediction")
            )
        ),
        {"class": "bslib-page-dashboard"},
    ),
    title="Volatility Prediction Dashboard",
    fillable=True,
    id="tabs"
)

# --- Server Logic ---
def server(input: Inputs):
   prediction_cache = {}

   @reactive.calc()
   def prediction_all():
        sid = input.selected_stock_id_forecast()
        tid = input.selected_time_id_forecast()
        cache_key = (sid, tid)

        if cache_key in prediction_cache:
            return prediction_cache[cache_key]

        # 👇 Define prediction output path
        pred_path = f"dashboard/predictions/pred_{sid}_tid{tid}.pkl"
        os.makedirs("dashboard/predictions", exist_ok=True)

        if os.path.exists(pred_path):
            df_result = pd.read_pickle(pred_path)
            prediction_cache[cache_key] = df_result
            return df_result

        # 🧠 Otherwise: run model inference
        model_path = "out/lstm/advanced.h5"
        lstm_model = load_model(model_path, custom_objects={
            "mse": mse,
            "qlike_loss": model.qlike_loss,
            "directional_accuracy": model.directional_accuracy
        })

        X, y_true, time_ids, start_times = model.prepare_lstm_data(sid, tid)
        y_pred = lstm_model.predict(X, verbose=0).ravel()

        df_result = pd.DataFrame({
            "time_id": time_ids,
            "start_time": start_times,
            "y_true": y_true,
            "y_pred": y_pred
        })

        # 💾 Save prediction to disk
        df_result.to_pickle(pred_path)

        prediction_cache[cache_key] = df_result
        return df_result


   
   @reactive.calc()
   def cached_predictions():
        return prediction_all()
   
   @render.text
   def qlike():
        df = cached_predictions()
        return qlike_loss(df["y_true"].values, df["y_pred"].values).numpy()
   
   @render.text
   def mse():
        df = cached_predictions()
        return float(mse(df["y_true"].values, df["y_pred"].values).numpy())
   
   @render.text
   def rmse():
        df = cached_predictions()
        error = mse(df["y_true"].values, df["y_pred"].values)
        return float(rmse(error.numpy()))  # `rmse` expects a scalar

   
   @render.plot
   def predict_plot():
        df = cached_predictions()

        ax = sns.lineplot(x=range(len(df)), y=df["y_true"], label="Actual", linestyle="-", color="blue")
        sns.lineplot(x=range(len(df)), y=df["y_pred"], label="Predicted", linestyle="--", color="orange", ax=ax)
        ax.set_title("Actual vs Predicted Volatility")
        ax.set_xlabel("Time")
        ax.set_ylabel("Realized Volatility")
        return ax.figure

   
   
#    @render.plot
#    def display_prediction():
#         selected_ids = input.stock_ids()
#         if not selected_ids:
#             return None  # Avoid plotting if none selected

#         all_predictions = cached_predictions()

#         predicted_dict = all_predictions["predicted"]

#         ax = None
#         for sid in selected_ids:
#             if sid not in predicted_dict:
#                 continue  # skip any unknown ID (defensive)

#             predicted = predicted_dict[sid]
#             label = f"Stock {sid}"

#             ax = sns.lineplot(
#                 x=range(len(predicted)),
#                 y=predicted,
#                 label=label,
#                 linestyle="--",
#                 ax=ax
#             )

#         ax.set_title("Predicted Volatility for Selected Stocks")
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Predicted Volatility")
#         return ax.figure


# --- Create App ---
app = App(app_ui, server)

