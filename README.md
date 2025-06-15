# Time-Series-Analysis-of-Ethereum-ETH-USDT-Market-Projections-using-ARIMA
A time series forecasting project using ARIMA to analyze and predict Ethereum (ETH/USDT) prices. Includes data collection, EDA, stationarity testing, ACF/PACF analysis, model evaluation, and 30-day future forecasting with visualizations. Dataset spans 2020–2025 with daily intervals. 
# 📊 Ethereum (ETH/USDT) Time Series Forecasting using ARIMA

This project performs time series analysis and forecasting of Ethereum (ETH/USDT) daily prices using the ARIMA model. It involves data collection, visualization, stationarity testing, model building, evaluation, and forecasting.

---

## 🧠 Project Overview

**Goal:** Analyze Ethereum's historical price trends and forecast future values using ARIMA (AutoRegressive Integrated Moving Average).

**Steps Covered:**
1. Data Collection & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Stationarity Testing (ADF)
4. ACF & PACF Plotting
5. ARIMA Model Development
6. Model Evaluation
7. Forecasting Next 30 Days

---

## 📦 Requirements

Install required libraries using pip:

pip install yfinance pandas numpy matplotlib seaborn statsmodels scikit-learn

## 📂 File Structure

├── task2.py                     # Main Python script (all 6 stages)
├── Cleaned_ETH_Data.csv         # Cleaned historical data
├── ETH_Forecast_30_Days.csv     # Forecasted prices for next 30 days
├── README.md                    # Project guide and documentation
└── .gitignore                   # Python-specific ignores (e.g., __pycache__)

## 🛠️ How to Run

Clone the repo or download the files.

Open task2.py in your IDE or run in terminal:
python task2.py
Visualizations and metrics will be displayed.
Forecast is saved in ETH_Forecast_30_Days.csv.

## 📉 Model Summary
Model: ARIMA(1, 1, 1)

Evaluation:

RMSE: Measures average error magnitude.

MAPE: Percent error on average prediction.

Forecasting: Plots Ethereum's expected prices for the next 30 days with confidence intervals.

## 📊 Data Source
Yahoo Finance

Ticker: ETH-USD

Date Range: 2020-01-01 to 2025-01-01

## 📘 License
This project is open-source under the MIT License. See LICENSE for more information.

