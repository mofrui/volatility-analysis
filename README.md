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
From project root, run the following command
```bash
shiny run --reload dashboard/app.py  
```