import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import mplcursors
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

# ASX stock symbol
symbol = "bhp.AX"

# the date range for historical data
start_date = "1980-01-01" 
end_date = datetime.now().strftime("%Y-%m-%d")

# download historical stock data - yfinance
data = yf.download(symbol, start=start_date, end=end_date)
df = data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()

# time Series Analysis (ARIMA)
time_series_data = data["Close"]
model = ARIMA(time_series_data, order=(5, 1, 0))
model_fit = model.fit()
forecast_steps = 7
forecast = model_fit.forecast(steps=forecast_steps)
forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_steps)

# Moving Average
df["30-day MA"] = df["Close"].rolling(window=30).mean()

# Remove rows with NaN values in the "30-day MA" column
df = df.dropna(subset=["30-day MA"])

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# extract features and target variables
features = pd.DataFrame(scaled_features, columns=df.columns)
target_high = df["High"]
target_low = df["Low"]

# Split the data into training and testing sets
X_train, X_test, y_train_high, y_test_high, y_train_low, y_test_low = train_test_split(
    features, target_high, target_low, test_size=0.2, random_state=42
)

# Train the machine learning model (Random Forest Regressor)
model_high = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model_high.fit(X_train, y_train_high)

model_low = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model_low.fit(X_train, y_train_low)

# Evaluate the model performance on the testing set
y_pred_high = model_high.predict(X_test)
y_pred_low = model_low.predict(X_test)

mse_high = mean_squared_error(y_test_high, y_pred_high)
mse_low = mean_squared_error(y_test_low, y_pred_low)

print("Mean Squared Error (High Prices):", mse_high)
print("Mean Squared Error (Low Prices):", mse_low)

# Prepare future dates and features
last_data_point = df.iloc[[-1]].values
future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=7, closed='right')
future_features = pd.DataFrame(index=future_dates, columns=df.columns, data=np.repeat(last_data_point, len(future_dates), axis=0))

# Scale and transform the future features
future_features_scaled = scaler.transform(future_features)

# Make predictions for high and low prices
predictions_high = model_high.predict(future_features_scaled)
predictions_low = model_low.predict(future_features_scaled)

# Plot the graph with hoverable lines
fig, ax = plt.subplots()
df["High"].plot(ax=ax, label="Actual High Prices")
df["Low"].plot(ax=ax, label="Actual Low Prices")
plt.plot(future_dates, predictions_high, label="Predicted High Prices (RF)", linestyle="--")
plt.plot(future_dates, predictions_low, label="Predicted Low Prices (RF)", linestyle="--")
plt.plot(forecast_dates, forecast, label="Predicted Close Prices (ARIMA)", linestyle="--")
plt.plot(df.index, df["30-day MA"], label="30-day Moving Average", linestyle="--")

cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

plt.xlabel("Date")
plt.ylabel("Price (AUD)")
plt.title(f"ASX {symbol} Stock Price Prediction")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()