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
import joblib


# --- UI: Sidebar & Panels ---
forecast_sidebar = ui.sidebar(
    ui.input_select("selected_stock_id_forecast", "Choose a Stock ID:", {50200: "50200: SPY XNAS (Training Stock)", 104919: "104919: QQQ XNAS (Most Correlated)", 22771: "22771: NFLX XNAS (Least Correlated)"}, selected=50200),
    ui.input_select("selected_time_id_forecast", "Choose a Time ID:", {14: "14", 46:"46", 246:"246"}, selected=14),
    ui.markdown("**Note:** Each `time_id` represents one hour of trading data."),
    ui.input_select(
    "forecast_horizon",
    "Show Predictions For:",
    {   "full": "Full horizon",
        "from_cutoff": "🔍 Zoomed Forecast"
    },
    selected="from_cutoff"
    ),
    ui.input_radio_buttons(
        "highlight_range",
        "Highlight Window:",
        choices={
            "none": "No highlight",
            "30": "Next 30 seconds",
            "60": "Next 1 minute"
        },
        selected="60",
        inline=True
    ), 



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
                                choices=["RMSE", "QLIKE"],
                                selected="RMSE",
                                inline=True
                            ),  
                                ui.markdown("This boxplot shows the overall performance across the `time_id`s on the dashboard(14, 46, 246) for each stock"),
                                ui.output_plot("metric_box_plot")
                            )
                        )
                    ),
                    style="font-size: 0.8rem; margin-bottom: 0 rem; margin-top: 0 rem;"
                ),

                ui.card(
                    ui.card_header("Actual vs Predicted Volatility"),
                    ui.output_plot("predict_plot")

                ),
                style="max-height: 85vh; overflow-y: auto; padding-right: 8px;"
            )

            )
    ),
    ui.nav_panel(
    "📊 Quoting Strategies",
     ui.layout_sidebar(
                     ui.sidebar(
                         ui.input_select("spread_stock_id", "Choose a Stock ID:", {
                             50200: "50200: SPY XNAS",
                             104919: "104919: QQQ XNAS"
                         }, selected=50200),
                         ui.input_numeric("spread_time_id", "Enter Time ID:", value=14, min=0),
                        ui.div(
                            ui.output_text("quote_error_msg"),
                            style="color: red; font-size: 0.85rem; margin-top: -0.25rem; margin-bottom: 0.5rem;"
                        )
                     ),
                     ui.div(
                         ui.card(
                             ui.card_header("Actual vs Predicted Bid-Ask Spread"),
                             ui.output_plot("bid_ask_forecast_plot")
                         ),
                         ui.card_header("📈 Predicted Quotes"),
                         ui.layout_columns(
                             ui.value_box("Mid Price (t+1)", ui.output_text("pred_mid_card")),
                             ui.value_box("Spread (t+1)", ui.output_text("pred_spread_card")),
                             ui.value_box("Quoted Bid", ui.output_text("quoted_bid_card")),
                             ui.value_box("Quoted Ask", ui.output_text("quoted_ask_card"))
                         ),
                         ui.div(
                            ui.output_text("spread_change_note"),
                            style="background-color: #fff4d6; padding: 1rem; border-left: 6px solid #ffa500; border-radius: 8px; margin-bottom: 1.2rem; font-size: 1.05rem; font-weight: 500; color: #333;"
                        ),
                         ui.card_header("📉 Current Market Quotes"),
                         ui.layout_columns(
                             ui.value_box("Current Mid Price", ui.output_text("real_mid_card")),
                             ui.value_box("Current Spread", ui.output_text("real_spread_card")),
                             ui.value_box("Current Bid", ui.output_text("real_bid_card")),
                             ui.value_box("Current Ask", ui.output_text("real_ask_card"))
                         )

                     )
                 )
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
            46: 2870,
            246: 1080,
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
        ax.legend(
            loc="upper left",           # Always top-right inside the plot
            bbox_to_anchor=(0, 1),       # Fine-tune anchor position (1,1) = top-right corner
            frameon=True,
            fontsize="small"
        )


        return ax.figure


    
    @render.plot
    def metric_box_plot():
        stock_ids = {
            50200: "SPY",
            104919: "QQQ",
            22771: "NFLX"
        }
        time_ids = [14, 246, 46]  # Add more if you want
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

        df_plot = pd.DataFrame(rows)
        selected_df = df_plot[df_plot["Metric"] == metric_name]

         # Highlight best performer (lowest value)
        stock_means = selected_df.groupby("Stock")["Value"].mean()
        best_stock = stock_means.idxmin()

        # Add emoji to name
        selected_df["Label"] = selected_df["Stock"].apply(
            lambda s: (
                f"★  SPY\n(Training)" if s == best_stock and s == "SPY" else
                f"★  QQQ\n(Most Correlated)" if s == best_stock and s == "QQQ" else
                f"★  NFLX\n(Least Correlated)" if s == best_stock and s == "NFLX" else
                "SPY\n(Training)" if s == "SPY" else
                "QQQ\n(Most Correlated)" if s == "QQQ" else
                "NFLX\n(Least Correlated)"
            )
        )

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

    @render.plot
    def bid_ask_forecast_plot():
        from spread_model import load_spread_model, load_precomputed_features
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        import ast

        sid = int(input.spread_stock_id())
        model = load_spread_model()

        # Map stock_id to corresponding CSV file
        stock_to_file = {
            50200: "dashboard/data/predictions_spy.csv",
            104919: "dashboard/data/predictions_qqq.csv"
        }

        pred_path = stock_to_file.get(sid)
        pred_df = load_precomputed_features(pred_path)

        # ✅ Load preprocessed snapshot features instead of recomputing
        spread_feature_path = f"dashboard/data/spread_features_{sid}_alltimeid.pkl"
        agg_df = pd.read_pickle(spread_feature_path)

        # --- Clean and merge ---
        pred_df["predicted_volatility_lead1"] = pred_df["predicted_value"].apply(
            lambda x: ast.literal_eval(x)[0] if pd.notna(x) else np.nan
        )
        pred_df["time_id"] = pred_df["time_id"].astype(int)
        pred_df["bucket_id_330s"] = pred_df["bucket_id_330s"].astype(int)
        agg_df["time_id"] = agg_df["time_id"].astype(int)
        agg_df["bucket_id_330s"] = agg_df["bucket_id_330s"].astype(int)

        merged_df = pd.merge(pred_df, agg_df, on=["time_id", "bucket_id_330s"], how="inner")

        # Drop rows with missing columns
        merged_df = merged_df.dropna(subset=[
            "predicted_volatility_lead1", "spread_pct", "realized_volatility",
            "wap", "imbalance", "depth_ratio", "log_return",
            "bid_ask_spread", "bid_ask_spread_lead1"
        ])

        # Scale features
        scaler = StandardScaler()
        scaled = scaler.fit_transform(merged_df[["spread_pct", "realized_volatility"]])
        merged_df["spread_pct_scaled"] = scaled[:, 0]
        merged_df["realized_volatility_scaled"] = scaled[:, 1]

        # Final input to model
        feature_cols = [
            "predicted_volatility_lead1", "spread_pct_scaled", "realized_volatility_scaled",
            "wap", "imbalance", "depth_ratio", "log_return", "bid_ask_spread"
        ]
        merged_df[feature_cols] = merged_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        merged_df = merged_df.dropna(subset=feature_cols + ["bid_ask_spread_lead1"])

        # Predict
        y_true = merged_df["bid_ask_spread_lead1"].values
        y_pred = model.predict(merged_df[feature_cols])

        merged_df["label"] = merged_df["time_id"].astype(str) + "_" + merged_df["bucket_id_330s"].astype(str)

        # Plot
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(merged_df["label"], y_true, label="Actual", color="blue", linewidth=1, alpha=0.8)
        ax.plot(merged_df["label"], y_pred, label="Predicted", color="orange", linestyle="--", linewidth=1, alpha=0.8)

        ax.set_title("Actual vs. Predicted Bid-Ask Spread", fontsize=14)
        ax.set_xlabel("Time Window (time_id_bucket)", fontsize=12)
        ax.set_ylabel("Bid-Ask Spread", fontsize=12)

        tick_step = len(merged_df["label"]) // 20
        ax.set_xticks(merged_df["label"][::tick_step])
        ax.tick_params(axis='x', rotation=45)

        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

        import joblib

    # Load models once
    mid_model = joblib.load("dashboard/Models/mid_price_model.pkl")
    spread_model = joblib.load("dashboard/Models/bid_ask_spread_model.pkl")

   
    @reactive.calc()
    def quote_prediction():
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        sid = int(input.spread_stock_id())
        tid = int(input.spread_time_id())

        # Load preprocessed spread feature file
        feature_path = f"dashboard/data/spread_features_{sid}_alltimeid.pkl"
        if not os.path.exists(feature_path):
            print(f" Feature file not found: {feature_path}")
            return {
                "pred_mid": np.nan, "pred_spread": np.nan,
                "bid": np.nan, "ask": np.nan,
                "real_mid": None, "real_spread": None
            }

        df = pd.read_pickle(feature_path)
        df = df[df["time_id"] == tid].copy()

        if df.empty:
            print(f" No data for stock {sid}, time_id {tid}")
            return {
                "pred_mid": np.nan, "pred_spread": np.nan,
                "bid": np.nan, "ask": np.nan,
                "real_mid": None, "real_spread": None,
                "error": f"No data for stock {sid}, time ID {tid}. Please try another time ID."
            }

        # Pick one row and its next row
        row = df.iloc[0:1].copy()
        next_row = df.iloc[1] if len(df) > 1 else None

        row["predicted_volatility_lead1"] = row["realized_volatility"]

        # Scale 3 features
        scaler = StandardScaler()
        row[["spread_pct_scaled", "realized_volatility_scaled", "predicted_volatility_lead1"]] = scaler.fit_transform(
            row[["spread_pct", "realized_volatility", "predicted_volatility_lead1"]]
        )

        mid_features = ['spread_pct', 'imbalance', 'depth_ratio', 'bid_ask_spread', 'realized_volatility']
        spread_features = ['predicted_volatility_lead1', 'spread_pct_scaled', 'realized_volatility_scaled',
                        'wap', 'imbalance', 'depth_ratio', 'log_return', 'bid_ask_spread']

        pred_mid = mid_model.predict(row[mid_features])[0]
        pred_spread = spread_model.predict(row[spread_features])[0]
        bid = pred_mid - pred_spread / 2
        ask = pred_mid + pred_spread / 2

        return {
            "pred_mid": pred_mid,
            "pred_spread": pred_spread,
            "bid": bid,
            "ask": ask,
            "real_mid": next_row["mid_price"] if next_row is not None else None,
            "real_spread": next_row["bid_ask_spread"] if next_row is not None else None
        }


    @render.text
    def pred_mid_card():
        q = quote_prediction()
        return f"{q['pred_mid']:.6f}"

    @render.text
    def pred_spread_card():
        q = quote_prediction()
        return f"{q['pred_spread']:.6f}"

    @render.text
    def quoted_bid_card():
        q = quote_prediction()
        return f"{q['bid']:.6f}"

    @render.text
    def quoted_ask_card():
        q = quote_prediction()
        return f"{q['ask']:.6f}"

    @render.text
    def real_mid_card():
        q = quote_prediction()
        return f"{q['real_mid']:.6f}" if q['real_mid'] is not None else "N/A"

    @render.text
    def real_spread_card():
        q = quote_prediction()
        return f"{q['real_spread']:.6f}" if q['real_spread'] is not None else "N/A"

    @render.text
    def real_bid_card():
        q = quote_prediction()
        if q['real_mid'] is None or q['real_spread'] is None:
            return "N/A"
        return f"{q['real_mid'] - q['real_spread'] / 2:.6f}"

    @render.text
    def real_ask_card():
        q = quote_prediction()
        if q['real_mid'] is None or q['real_spread'] is None:
            return "N/A"
        return f"{q['real_mid'] + q['real_spread'] / 2:.6f}"

    @render.text
    def spread_change_note():
        q = quote_prediction()
        if q['real_spread'] is None or np.isnan(q['real_spread']):
            return "📉 Spread trend info unavailable."
        change = q['pred_spread'] - q['real_spread']
        emoji = "⬆️" if change > 0 else "⬇️" if change < 0 else "➡️"
        direction = "increase" if change > 0 else "decrease" if change < 0 else "stay the same"
        return f"{emoji} Bid-ask spread is expected to <{direction}> in the next time id (t+1)."
    
    @render.text
    def quote_error_msg():
        q = quote_prediction()
        return q["error"] if "error" in q else ""


# --- Create App ---
app = App(app_ui, server)
