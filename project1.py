import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# =========== 1. Data Collection ==========
eth_data = yf.download('ETH-USD', start='2020-01-01', end='2025-01-01', interval='1d')
eth_data = eth_data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
eth_data.index = pd.to_datetime(eth_data.index)
eth_data.to_csv("Cleaned_ETH_Data.csv")

# ========== 2. Visualization ===========
sns.set(style='darkgrid')

# Close Price Over Time
plt.figure(figsize=(14, 6))
plt.plot(eth_data['Close'], label='Close Price')
plt.title('Ethereum (ETH-USD) Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Moving Averages
eth_data['7-day MA'] = eth_data['Close'].rolling(window=7).mean()
eth_data['30-day MA'] = eth_data['Close'].rolling(window=30).mean()

plt.figure(figsize=(14, 6))
plt.plot(eth_data['Close'], label='Close Price', alpha=0.5)
plt.plot(eth_data['7-day MA'], label='7-Day MA')
plt.plot(eth_data['30-day MA'], label='30-Day MA')
plt.title('ETH-USD Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Volume Over Time
plt.figure(figsize=(14, 4))
plt.plot(eth_data['Volume'], label='Volume', color='orange')
plt.title('Ethereum Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()

# ========== 3. Statistical Summary ==========
print("\n📊 Statistical Summary:")
print(eth_data.describe())

# ========== 4. Stationarity Testing ==========
def adf_test(series, title=''):
    print(f"\n📉 Augmented Dickey-Fuller Test: {title}")
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF Statistic', 'p-value', '# Lags Used', 'Num Observations Used']
    for val, label in zip(result, labels):
        print(f"{label}: {val}")
    if result[1] <= 0.05:
        print("✅ The data is stationary (reject H0).")
    else:
        print("⚠️ The data is non-stationary (fail to reject H0).")

# ADF on original Close price
adf_test(eth_data['Close'], title='Original Closing Price')

# Differencing to make stationary
eth_data['Close_diff'] = eth_data['Close'].diff()
adf_test(eth_data['Close_diff'], title='First Differenced Closing Price')

# Plot Differenced Series
plt.figure(figsize=(12, 6))
plt.plot(eth_data['Close_diff'], color='green')
plt.title('Differenced Closing Price (ETH)')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.show()

# ========== 5. ACF and PACF ==========
diff_series = eth_data['Close_diff'].dropna()

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(diff_series, ax=axes[0], lags=40)
axes[0].set_title('ACF - Differenced Close')
plot_pacf(diff_series, ax=axes[1], lags=40, method='ywm')
axes[1].set_title('PACF - Differenced Close')
plt.tight_layout()
plt.show()

# ========== 6. ARIMA Modeling and Forecasting ==========
model = ARIMA(eth_data['Close'], order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# Residual Plot
plt.figure(figsize=(10, 4))
plt.plot(model_fit.resid)
plt.title('Residuals from ARIMA(1,1,1)')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.show()

# Evaluation
y_true = eth_data['Close'].iloc[1:]  # drop the first NaN after differencing
y_pred = model_fit.fittedvalues
y_true, y_pred = y_true.align(y_pred, join='inner', axis=0)  # ✅ Fix applied here

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"\n📈 Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape * 100:.2f}%")


# ========== 7. Forecast Next 30 Days ==========
forecast = model_fit.get_forecast(steps=30)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

plt.figure(figsize=(12, 5))
plt.plot(eth_data['Close'], label='Historical Closing Price')
plt.plot(forecast_mean.index, forecast_mean, label='Forecast (Next 30 Days)', color='green')
plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='lightgreen', alpha=0.5)
plt.title('ETH-USD Forecast for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# Save forecast
forecast_mean.to_csv("ETH_Forecast_30_Days.csv")
