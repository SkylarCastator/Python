import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas_datareader as web
import numpy as np
from mpl_finance import candlestick_ohlc

start = dt.datetime(2019,1,1)
end = dt.datetime.now()
df = web.DataReader('AAPL', 'yahoo', start, end)
#print(df.head())
df = df[['Open', 'High', 'Low', 'Close']]
print(df['Close'][0])
df.reset_index(inplace=True)
df['Date'] = df['Date'].map(mdates.date2num)

ax = plt.subplot()

df["20d"] = np.round(df["Close"].rolling(window = 20, center = False).mean(), 2)
plt.plot(df['Date'], df["20d"], label="AAPL")
df["50d"] = np.round(df["Close"].rolling(window = 50, center = False).mean(), 2)
plt.plot(df['Date'], df["50d"], label="AAPL")
df["200d"] = np.round(df["Close"].rolling(window = 200, center = False).mean(), 2)
plt.plot(df['Date'], df["200d"], label="AAPL")
 

candlestick_ohlc(ax, df.values, width=5, colorup='g', colordown='r')
ax.grid(True)
ax.set_axisbelow(True)
ax.set_title('AAPL Share Price', color='white')
ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.xaxis_date()
plt.show()
