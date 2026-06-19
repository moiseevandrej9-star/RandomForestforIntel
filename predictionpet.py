import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

ticker = "INTC"
data = yf.download(ticker, start="2023-01-01", end="2026-06-01")

# Лаги - значения за прошлые дни как признаки
for lag in [1, 2, 3, 5, 10]:
    data[f'lag_{lag}'] = data['Close'].shift(lag)

data['MA_10'] = data['Close'].rolling(window=10).mean().shift(1)
data['Volatility'] = data['Close'].rolling(window=10).std().shift(1)

data = data.dropna()

features = ['lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10', 'MA_10', 'Volatility', 'momentum']
data['momentum'] = data['Close'].diff(5).shift(1)
X = data[features]
y = data['Close']

split = int(len(data) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("R2 score:", r2_score(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))

plt.figure(figsize=(10,6))
plt.plot(range(len(y_test)), y_test.values, label='Реальная цена')
plt.plot(range(len(y_test)), predictions, label='Прогноз', linestyle='--')
plt.xlabel('День (тест)')
plt.ylabel('Цена закрытия')
plt.title(f'Прогноз цены акции {ticker} (lag features)')
plt.legend()
plt.savefig('prediction_plot_lag.png')
plt.show()