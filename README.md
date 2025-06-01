# Precision Volatility Forecasting for Strategic Quote Placement in High-Frequency Trading

## Overview

This repository contains the code and report for the DATA3888 Interdisciplinary Project (Optiver Stream).  
The project focuses on precision volatility forecasting to support strategic quoting in high-frequency trading (HFT).

The main goals are:
1. Develop models to predict short-term volatility of a stock.
2. Leverage these forecasts to determine optimal bid-ask quote placement.

## Research Question

> **How can short-term volatility forecasts be leveraged to optimize bid-ask spread quoting strategies, balancing execution risk and market competitiveness, for HFT firms?**  
> Additionally, how does inter-stock correlation influence both prediction accuracy and quoting effectiveness?

## Setup Instructions

1. **Navigate to the project root directory:**
    ```bash
    cd optiver22
    ```

2. **Run the setup script to install all required Python dependencies:**
    ```bash
    bash install.sh
    ```
    ⚠️ This project requires **Python 3.11.x**. The script will exit if you’re using another version.


## File structure

```plaintext
├── dashboard/      # Interactive Shiny App for visualizing predictions
├── models/         # Trained models (e.g., LSTM .h5, scalers .pkl)
├── resources/      # Figures, diagrams, and additional project assets
├── src/            # Source code for modeling
├── temp/           # Temporary and intermediate data
├── report.qmd      # Main Quarto report (compiled to HTML)
├── report.html     # Compiled report (HTML)
├── install.sh      # Installation script for dependencies
├── .gitignore      # Git ignore file
├── .python-version # Python version configuration (e.g., for pyenv)
├── README.md       # Project overview and instructions
```

## Running dashboard 

1. **(Skip if already done)** Make sure you have run the setup script from the project root to install all required Python packages:
```bash
bash install.sh
```
2. run the shiny app using 
```bash
python3 -m shiny run --reload dashboard/app.py
```

##  File Structure for Running the Dashboard

The following files and folders are included in the `dashboard/` directory and are required for the app to run:
```output
dashboard/
├── app.py                  # Main Shiny app entry point
├── model.py                # Core model loading and prediction logic
├── spread_model.py         # For bid-ask spread prediction
├── data/                   # Preprocessed input files (.pkl)
│   └── [e.g., 50200_tid14.pkl, 104919_tid246.pkl, ...]
├── Models/                 # Contains trained models
│   ├── bid_ask_spread_model.pkl
│   └── mid_price_model.pkl
|   ├── final.h5
│   └── final_scalers.pkl
├── predictions/            # Saved prediction outputs
│   └── [e.g., pred_50200_14.pkl, pred_104919_46.pkl, ...]
```
 > Note: If the `predictions/` folder is missing, it will be automatically created during app execution.


## Generating the Report

To generate the fully reproducible report (`report.html`):

```bash
quarto render report.qmd
```

Please ensure you have the `data\` directory in the root folder with `order_book_feature.parquet`, `order_book_target.parquet`, `stock_id.csv` and `time_id_reference.csv` files, othervise the report will not run correctly.

The report will re-run all Python code blocks, ensuring that all results and figures are updated.

If you want to generate the report without running the model training and predictions (only use precomputed `temp` files), please change the `BUILD_MODEL` and `RUN_EVALUATION` parameter in the first code chunks to `False`.
