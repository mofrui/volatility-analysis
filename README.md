# Precision Volatility Forecasting for Strategic Quote Placement in High-Frequency Trading
## Overview

This repository contains the code and report for the DATA3888 Interdisciplinary Project (Optiver Strean).
The project focuses on precision volatility forecasting for strategies quote placement in high-frequency trading.

The goal of this project is to;
1. Develop a model to predict the volatility of a stock in the future.
2. Use the predicted volatility to determine the optimal quote placement for a stock.

## Research Question
How can short-term volatility forecasts be leveraged to optimize bid-ask spread quoting strategies, balancing execution risk and market competitiveness, for HFT firms, and how does inter-stock correlation influence both prediction accuracy and quoting effectiveness.


## 🚀 Setup Instructions
1. Make sure you are in project folder 
```bash
cd volatility-analysis
```
This project requires **Python 3.11.x** or later and tensorflow==2.16.2 
2. Run the setup script (installs Python packages)
```bash
bash install.sh
```
⚠️ This project requires Python 3.11. The script will exit if you're using another version.

## File structure 
>[!WARNING]
> based on folder that we are gonna submit - report, scritps to generate report, dashboard


## Running dashboard 
1. make sure that you are in `dashboard` directory 
```bash
cd dashboard
```
2. run the shiny app using 
```bash
shiny run --reload app.py  
```

##  File Structure for Running the Dashboard
The following files and folders are included in the dashboard/ directory and are required for the app to run:
```bash
dashboard/
├── app.py                   # Main Shiny app entry point
├── model.py                # Core model loading and prediction logic
├── spread_model.py         # For bid-ask spread prediction
├── data/                   # Preprocessed input files (.pkl)
│   └── [e.g., 50200_tid14.pkl, 104919_tid246.pkl, ...]
├── Models/                 # Contains trained spread models
│   ├── bid_ask_spread_model.pkl
│   └── mid_price_model.pkl
├── out/
│   └── lstm/               # LSTM model and Corresponding scalers
│       ├── moe_staged_full.h5
│       └── moe_staged_scalers_full.pkl
├── predictions/            # Saved prediction outputs
│   └── [e.g., pred_50200_14.pkl, pred_104919_46.pkl, ...]
```
 > Note: If the predictions/ folder is missing, it will be automatically created during app execution.


## Literatures

>[!WARNING]
> Please add more literatures to this section for later literature reviews.

> Yuqing Feng, Yaojie Zhang. 2024. *Forecasting Realized Volatility: The Choice of Window Size*. Journal of Forcasting (2025). https://doi.org/10.1002/for.3221

> Akgun, O.B., Gulay, E. *Dynamics in Realized Volatility Forecasting: Evaluating GARCH Models and Deep Learning Algorithms Across Parameter Variations*. Comput Econ (2024). https://doi.org/10.1007/s10614-024-10694-2

## Methodology

### Data Preprocessing and Feature Engineering

#### Dataset at a Glance

We use high-frequency order book data from 10 different stocks, each split by hour (`time_id`).
Every second within each hour is recorded, providing 1800 snapshots per stock per hour.
Our focus is on predicting short-term realized volatility using features derived from these snapshots.

#### Feature Engineering

From the order book, we compute features:

- Weighted Average Price (WAP)
- Bid-Ask Spread
- Order Imbalance
- Log Returns
- ...

### Pipeline 1: Forcasting Realised Volatility

We apply a rolling window approach to better suit the HFT context.

#### Determine the Rolling Window Size

We experiment with various rolling window configurations:

- Window size (`W`): 30s, 60s, 120s
- Forecast horizon (`H`): 30s, 60s, 120s
- Step size (`S`): 10s, 20s

We evaluate the performance of each configuration using a baseline model (e.g. Random Forest) to determine which window setting provides the most accurate and stable volatility predictions.

#### Model Training and Validation

We construct a rolling window dataset using 8 out of the 10 stocks for training.
The remaining 2 stocks are used for testing the model's generalization ability across correlated and uncorrelated assets.

To do this, we first compute the average log returns for each stock over time and calculate a correlation matrix.
Based on this analysis:

- Stock B is chosen as the most correlated with the training set.
- Stock C is chosen as the least correlated.

This allows us to test both in-domain and out-of-domain generalization, and the robustness of the model in an unseen stock.

We train and evaluate the following models using the best-performing rolling window setup.

Suggestion: Instead of splitting by stock (e.g. 8 train, 2 test), train and test on different time periods within the same stock.​
 For example we could do: ​

Use time_id_reference.csv to map time_ids to real time.​
 ➤ Train on one day → Test on the next day​
 ➤ Train on one week → Test on the following week ​


##### Baseline: HAR-RV Model

>[!WARNING]
> TODO

##### ARMA-GARCH Model (Various Variations)

Explore multiple ARMA(p,q)-GARCH(1,1) combinations, selecting the best based on performance.

##### LSTM Model (With Hyperparameter Tuning)

>[!WARNING]
> TODO

##### XGBoost Model (With Hyperparameter Tuning)

>[!WARNING]
> TODO

#### Movel Selection and Evaluation

We compare model performance using three perspectives:

Metrics Analysis: `MSE` and `QLIKE`.

Trend Cathing Analysis: We visually compare predicted vs. actual volatility to assess whether the model captures spikes and directional changes.

Robustness Analysis: We test whether the selected model generalizes well to unseen stocks: Stock B (highly correlated with training set) and Stock C (low or uncorrelated with training set)

### Pipeline 2: Predicting the Optimal Quote Placement

Objective: Use predicted volatility to decide whether to widen or tighten the bid-ask spread

Classify market conditions into:

- High volatility = widen spread
- Low volatility = tighten spread

>[!WARNING]
> TODO
