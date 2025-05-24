# dashboard.py (cleaned)
from shiny import App, Inputs, reactive, render, ui
import seaborn as sns
import pandas as pd
import model
from model import load_model, qlike_loss, mse_custom, rmse_custom
import pickle
import time
import os
import numpy as np


# --- UI: Sidebar & Panels ---
forecast_sidebar = ui.sidebar(
    ui.input_select("selected_stock_id_forecast", "Choose a Stock ID:", {50200: "50200: SPY XNAS", 104919: "104919: QQQ XNAS", 22771: "22771: NFLX XNAS"}, selected=50200),
    ui.input_select("selected_time_id_forecast", "Choose a Time ID:", {14: "14", 46:"46", 54:"54", 246:"246"}, selected=14),
    ui.markdown("**Note:** Each `time_id` represents one hour of trading data."),
    ui.input_select(
    "forecast_horizon",
    "Show Predictions For:",
    {   "full": "Full forecast",
        "from_cutoff": "🔍 Zoomed Forecast"
    },
    selected="full"
)



)

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

    ui.nav_panel("📊 Volatility Forecast",
        ui.layout_sidebar(
            forecast_sidebar,
            ui.div(
                ui.div(  # Compact tab styling
                    ui.navset_tab(
                        ui.nav_panel("🔢 Metric Summary",
                            ui.layout_columns(
                                ui.value_box(title="Q-like", value=ui.output_text("qlike")),
                                ui.value_box(title="RMSE", value=ui.output_text("rmse")),
                                ui.value_box(title="Inference Time", value=ui.output_text("inference_time"))
                            )
                        ),
                        ui.nav_panel("Metric Box Plot",
                            ui.card(
                                ui.card_header("Metric Comparison by Stock"),
                                ui.input_radio_buttons(
                                "metric_to_plot",
                                label=None,
                                choices=["RMSE", "QLIKE", "Inference Time (s)"],
                                selected="RMSE",
                                inline=True
                            ),  
                                ui.markdown("This box plot shows overall performance (e.g., RMSE, QLIKE) across multiple `time_id`s for each stock."),
                                ui.output_plot("metric_box_plot")
                            )
                        )
                    ),
                    style="font-size: 0.8rem; margin-bottom: 0 rem; margin-top: 0 rem;"
                ),

                ui.card(
                    ui.card_header("Actual vs Predicted Volatility"),
                    ui.input_radio_buttons(
                    "highlight_range",
                    "Highlight Window:",
                    choices={
                        "none": "No highlight",
                        "30": "Next 30 seconds",
                        "60": "Next 1 minute"
                    },
                    selected="none",
                    inline=True
                    ), 
                    ui.output_plot("predict_plot")

                ),
                style="max-height: 85vh; overflow-y: auto; padding-right: 8px;"
            )

            )
    ),
    ui.nav_panel(
    "📊 Quoting Strategies",
    ui.layout_sidebar(
        ui.sidebar(  # <- REQUIRED: this is the actual sidebar (even if empty or minimal)
            ui.markdown("Use the predicted volatility below to guide quoting decisions.")
        ),
        ui.card(  # <- main content area
            ui.card_header("Volatility-Informed Quoting Strategy"),
            ui.output_plot("display_prediction")
        )
    ),
),


   
    title="Volatility Prediction Dashboard",
    fillable=True,
    id="tabs"
)

# --- Server Logic ---
def server(input: Inputs):
    prediction_cache = {}
    # Load model + scalers only once
    model_path = "out/lstm/moe_staged_full.h5"
    scaler_path = "out/lstm/moe_staged_scalers_full.pkl"


    lstm_model = load_model(model_path)
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]

    @reactive.calc()
    def prediction_all():
        sid = input.selected_stock_id_forecast()
        tid = input.selected_time_id_forecast()
        cache_path = f"dashboard/predictions/pred_{sid}_{tid}.pkl"

        # If already cached, load from .pkl
        if os.path.exists(cache_path):
            print(f"[CACHE] Loaded precomputed result from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        else:
            # Otherwise, compute and save
            print(f"[COMPUTE] Running prediction for stock {sid}, time {tid}")

         # start calc Total computation time (data + scaling + prediction)
        start_time = time.time()
        X, y_true, time_ids, start_times = model.prepare_lstm_data(sid, tid)
        X_scaled = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)  
        y_pred_outputs = lstm_model.predict(X_scaled, verbose=0)

        # If the model has multiple outputs, use the first (volatility)
        if isinstance(y_pred_outputs, list):
            y_pred_scaled = y_pred_outputs[0]
        else:
            y_pred_scaled = y_pred_outputs

        print("Prediction outputs type:", type(y_pred_outputs))
        print("Number of outputs:", len(y_pred_outputs) if isinstance(y_pred_outputs, list) else 1)
        print("Shape of first output:", np.array(y_pred_scaled).shape)

        y_pred_scaled = y_pred_scaled.reshape(-1, 1)


        y_pred_scaled = y_pred_scaled.reshape(-1, 1)  # Fix the 3D -> 2D shape
        y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()

        elapsed = time.time() - start_time
         # --- DEBUG PRINTS ---
        print("y_true range:", y_true.min(), y_true.max())
        print("y_pred range:", y_pred.min(), y_pred.max())

        print("X sample (before scaling):", X[0][0])
        print("X sample (after scaling):", X_scaled[0][0])
        print("y_true sample:", y_true[:5])
        print("y_pred_scaled sample:", y_pred_scaled[:5])
        print("y_pred final sample:", y_pred[:5])
        print("----")

        df = pd.DataFrame({"start_time": start_times, "y_true": y_true, "y_pred": y_pred})
        df.attrs["inference_time"] = elapsed  # attach as metadata

           # Save for future instant loading
        os.makedirs("dashboard/predictions", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
        print(f"[CACHE] Saved result to {cache_path}")

        return df


    @reactive.calc()
    def cached_predictions():
        return prediction_all()
    
    @render.text
    def qlike():
        df = cached_predictions()
        return f"{qlike_loss(df['y_true'], df['y_pred']):.6f}"

    @render.text
    def rmse():
        df = cached_predictions()
        return f"{rmse_custom(df['y_true'].values, df['y_pred'].values):.6f}"
    
    @render.text
    def inference_time():
        df = cached_predictions()
        return f"{df.attrs['inference_time']:.4f} seconds"


    @render.plot
    def predict_plot():
        df = prediction_all()
        tid = input.selected_time_id_forecast()

        # Define custom cutoff point per time_id
        cutoff_map = {
            14: 2230,
            46: 1450,
            54: 1600,
            246: 1700,
            # Add more if needed
        }
        cutoff_sec = cutoff_map.get(int(tid), 1000)  # fallback if not listed

        raw_horizon = input.forecast_horizon()

        # Define prediction window
        if raw_horizon == "full":
            df_plot = df.copy()
            end_cutoff = df["start_time"].max()
        elif raw_horizon == "from_cutoff":
            start_zoom = (cutoff_sec // 500) * 500
            # df_plot = df[df["start_time"] >= start_zoom].copy()
            # end_cutoff = df_plot["start_time"].max()

            end_zoom = ((cutoff_sec + 500) // 500) * 500  # pad to next 500 boundary
            df_plot = df[(df["start_time"] >= start_zoom) & (df["start_time"] <= end_zoom)].copy()
            end_cutoff = end_zoom
        else:
            horizon = int(raw_horizon)
            end_cutoff = cutoff_sec + horizon
            df_plot = df[(df["start_time"] >= cutoff_sec - 100) & (df["start_time"] <= end_cutoff)].copy()

        # Mask predictions before cutoff
        df_plot.loc[df_plot["start_time"] < cutoff_sec, "y_pred"] = float("nan")

        # Plot
        ax = sns.lineplot(x=df_plot['start_time'], y=df_plot['y_true'], label="Actual", color="#3d5a80")
        sns.lineplot(x=df_plot['start_time'], y=df_plot['y_pred'], label="Predicted", linestyle="--", color="#ee6c4d", ax=ax)
        ax.axvline(x=cutoff_sec, color='red', linestyle='-', label="Prediction Start")

        # Overlay for 30s & 1min windows
        highlight = input.highlight_range()

        if highlight == "30":
            ax.axvspan(cutoff_sec, cutoff_sec + 30, color="skyblue", alpha=0.15, label="Next 30s")
        elif highlight == "60":
            ax.axvspan(cutoff_sec, cutoff_sec + 60, color="orange", alpha=0.12, label="Next 1min")

        ax.set_title(f"Actual vs Predicted Volatility (Prediction begins at {cutoff_sec}s)")
        ax.set_xlabel("Seconds in Bucket")
        ax.set_ylabel("Realized Volatility")
        ax.legend()

        return ax.figure


    
    @render.plot
    def metric_box_plot():
        stock_ids = {
            50200: "SPY",
            104919: "QQQ",
            22771: "NFLX"
        }
        time_ids = [14, 246, 54, 46]  # Add more if you want
        metric_name = input.metric_to_plot()
        rows = []

        for tid in time_ids:
            for sid, name in stock_ids.items():
                cache_path = f"dashboard/predictions/pred_{sid}_{tid}.pkl"
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        df = pickle.load(f)
                else:
                    X, y_true, time_ids_arr, start_times = model.prepare_lstm_data(sid, tid)
                    X_scaled = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
                    y_pred_scaled = lstm_model.predict(X_scaled, verbose=0)
                    y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
                    df = pd.DataFrame({"start_time": start_times, "y_true": y_true, "y_pred": y_pred})
                    df.attrs["inference_time"] = time.time()

                qlike = qlike_loss(df["y_true"], df["y_pred"])
                rmse = rmse_custom(df["y_true"], df["y_pred"])
                t = df.attrs.get("inference_time", 0)

                rows.append({
                    "Stock": name,
                    "Metric": "QLIKE",
                    "Value": qlike
                })
                rows.append({
                    "Stock": name,
                    "Metric": "RMSE",
                    "Value": rmse
                })
                rows.append({
                    "Stock": name,
                    "Metric": "Inference Time (s)",
                    "Value": t
                })

        df_plot = pd.DataFrame(rows)
        selected_df = df_plot[df_plot["Metric"] == metric_name]

         # Highlight best performer (lowest value)
        stock_means = selected_df.groupby("Stock")["Value"].mean()
        best_stock = stock_means.idxmin()

        # Add emoji to name
        selected_df["Label"] = selected_df["Stock"].apply(lambda s: f"★  {s}" if s == best_stock else s)

        # Define custom color palette
        unique_labels = selected_df["Label"].unique()
        highlight_color = "#80f085"  
        default_color = "#5a84ba"    
        palette = {
            label: (highlight_color if "★" in label else default_color)
            for label in unique_labels
        }

        # Plot with custom label column
        ax = sns.boxplot(data=selected_df, x="Label", y="Value", palette=palette)
        ax.set_title(f"{metric_name} Across Stocks (Lower = Better)" if "QLIKE" in metric_name or "RMSE" in metric_name else f"{metric_name} Across Stocks")
        return ax.figure


# --- Create App ---
app = App(app_ui, server)
