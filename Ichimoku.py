# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Step 2: Fetch USD/JPY daily data from Yahoo Finance
ticker = 'USDJPY=X'
data = yf.download(ticker, start='2022-01-01', end='2024-08-31')
data = data[['Close']]  # Keep only the closing price
data.dropna(inplace=True)

# Step 3: Define a function to calculate the Ichimoku indicator components
def ichimoku(df):
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
    df['Senkou_span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_span_B'] = ((high_52 + low_52) / 2).shift(26)

    # Chikou Span (Lagging Span): Close plotted 26 periods in the past
    df['Chikou_span'] = df['Close'].shift(-26)

    return df

# Add high and low prices for Ichimoku calculation
data['High'] = data['Close'].rolling(window=9).max()
data['Low'] = data['Close'].rolling(window=9).min()

# Step 4: Calculate the Ichimoku components
data = ichimoku(data)

# Step 5: Define the trading logic
# Buy when the Tenkan-sen crosses above the Kijun-sen
# Sell when the Tenkan-sen crosses below the Kijun-sen

def trading_strategy(df):
    df['Signal'] = 0  # Default signal
    df['Signal'][df['Tenkan_sen'] > df['Kijun_sen']] = 1  # Buy signal
    df['Signal'][df['Tenkan_sen'] < df['Kijun_sen']] = -1  # Sell signal
    df['Position'] = df['Signal'].shift()  # Lag to avoid look-ahead bias
    df.dropna(inplace=True)
    return df

data = trading_strategy(data)

# Step 6: Calculate strategy returns
data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data['Strategy_Returns'] = data['Returns'] * data['Position']

# Step 7: Calculate cumulative returns
data['Cumulative_Market_Returns'] = data['Returns'].cumsum().apply(np.exp)
data['Cumulative_Strategy_Returns'] = data['Strategy_Returns'].cumsum().apply(np.exp)

# Step 8: Plot cumulative returns
plt.figure(figsize=(12, 8))
plt.plot(data['Cumulative_Market_Returns'], label='Market Returns (Buy & Hold)')
plt.plot(data['Cumulative_Strategy_Returns'], label='Strategy Returns (Ichimoku)')
plt.title('Cumulative Returns: Ichimoku vs Market (USD/JPY)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Backtesting: Visualize buy/sell signals on price chart
plt.figure(figsize=(14, 8))
plt.plot(data.index, data['Close'], label='Close Price', alpha=0.6)
plt.plot(data.index, data['Tenkan_sen'], label='Tenkan-sen (Conversion Line)', linestyle='--')
plt.plot(data.index, data['Kijun_sen'], label='Kijun-sen (Base Line)', linestyle='--')
plt.fill_between(data.index, data['Senkou_span_A'], data['Senkou_span_B'], 
                 where=data['Senkou_span_A'] >= data['Senkou_span_B'], color='lightgreen', alpha=0.3)
plt.fill_between(data.index, data['Senkou_span_A'], data['Senkou_span_B'], 
                 where=data['Senkou_span_A'] < data['Senkou_span_B'], color='lightcoral', alpha=0.3)
plt.scatter(data.index[data['Position'] == 1], 
            data['Close'][data['Position'] == 1], marker='^', color='g', label='Buy Signal', alpha=1)
plt.scatter(data.index[data['Position'] == -1], 
            data['Close'][data['Position'] == -1], marker='v', color='r', label='Sell Signal', alpha=1)
plt.title('Ichimoku Strategy: Buy/Sell Signals (USD/JPY)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()