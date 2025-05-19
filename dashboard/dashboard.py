from shiny import App, Inputs, reactive, render, ui
from faicons import icon_svg
import seaborn as sns
import statsmodels.api as sm
import model

# --- Sidebar: Stock selection ---
sidebar = ui.sidebar(
    ui.input_select(
        "selected_stock_id",
        "Choose a Stock ID you want to predict:",
        {
            22771: "22771: NFLX XNAS",
            104919: "104919: QQQ XNAS",
            50200: "50200: SPY XNAS",
        },
        selected=22771,
    ),
    ui.input_checkbox_group(
        "stock_ids",
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
            sidebar,  
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
            sidebar,  
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
    @reactive.calc()
    def prediction_all():
        model_path = "Models/ols_model.pkl"
        selected_model = model.load_model(model_path)

        stock_ids = ["22771", "104919", "50200"]  # All supported stock IDs

        actual_dict = {}
        predicted_dict = {}

        for sid in stock_ids:
            load_data = model.load_data(int(sid))
            X = load_data[["rv_lag_1", "rv_lag_5", "rv_lag_10"]]
            y = load_data["realized_volatility"]
            X_const = sm.add_constant(X)

            actual_dict[sid] = y.values
            predicted_dict[sid] = selected_model.predict(X_const)

        return {
            "actual": actual_dict,
            "predicted": predicted_dict
        }

    @render.text
    def qlike():
        sid = input.selected_stock_id()
        all_predictions = prediction_all()
        return model.qlike(all_predictions["actual"][sid], all_predictions["predicted"][sid])

    @render.text
    def mse():
        sid = input.selected_stock_id()
        all_predictions = prediction_all()
        return model.mse(all_predictions["actual"][sid], all_predictions["predicted"][sid])

    @render.text
    def rmse():
        sid = input.selected_stock_id()
        all_predictions = prediction_all()
        return model.rmse(all_predictions["actual"][sid], all_predictions["predicted"][sid])

    @render.plot
    def predict_plot():
        sid = input.selected_stock_id()
        all_predictions = prediction_all()

        actual = all_predictions["actual"][sid]
        predicted = all_predictions["predicted"][sid]

        ax = sns.lineplot(x=range(len(actual)), y=actual, label="Actual", linestyle="-", color="blue")
        sns.lineplot(x=range(len(predicted)), y=predicted, label="Predicted", linestyle="--", color="orange", ax=ax)
        ax.set_title(f"Actual vs Predicted Volatility for Stock {sid}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Realized Volatility")
        return ax.figure

    @render.plot
    def display_prediction():
        selected_ids = input.stock_ids()
        if not selected_ids:
            return None  # Avoid plotting if none selected

        all_predictions = prediction_all()
        predicted_dict = all_predictions["predicted"]

        ax = None
        for sid in selected_ids:
            if sid not in predicted_dict:
                continue  # skip any unknown ID (defensive)

            predicted = predicted_dict[sid]
            label = f"Stock {sid}"

            ax = sns.lineplot(
                x=range(len(predicted)),
                y=predicted,
                label=label,
                linestyle="--",
                ax=ax
            )

        ax.set_title("Predicted Volatility for Selected Stocks")
        ax.set_xlabel("Time")
        ax.set_ylabel("Predicted Volatility")
        return ax.figure


# --- Create App ---
app = App(app_ui, server)