import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas_datareader as web
import numpy as np
from mpl_finance import candlestick_ohlc

start = dt.datetime(2019,1,1)
end = dt.datetime.now()
df = web.DataReader('AAPL', 'yahoo', start, end)
df = df[['Open', 'High', 'Low', 'Close']]
df.reset_index(inplace=True)
df['Date'] = df['Date'].map(mdates.date2num)
dft0_earnings = []
for i in df['Close']: 
    dft0_earnings.append(i / df['Close'][0])
dfl0_logDiff = []
for i in df['Close']: 
    if i==0:
         dfl0_logDiff.append(0)
    else:
         dfl0_logDiff.append(np.log(np.round(i)) - np.log(np.round(df['Close'][i-1])))

df1 = web.DataReader('GOOG', 'yahoo', start, end)
df1 = df1[['Open', 'High', 'Low', 'Close']]
df1.reset_index(inplace=True)
df1['Date'] = df1['Date'].map(mdates.date2num)
dft1_earnings = []
for i in df1['Close']: 
    dft1_earnings.append(i / df1['Close'][0])
dfl1_logDiff = []
for i in df1['Close']: 
    if i==0:
         dfl1_logDiff.append(0)
    else:
         dfl1_logDiff.append(np.log(i) - np.log(df1['Close'][i-1]))

df2 = web.DataReader('MSFT', 'yahoo', start, end)
df2 = df2[['Open', 'High', 'Low', 'Close']]
df2.reset_index(inplace=True)
df2['Date'] = df2['Date'].map(mdates.date2num)
dft2_earnings = []
for i in df2['Close']: 
    dft2_earnings.append(i / df2['Close'][0])
dfl2_logDiff = []
for i in df2['Close']: 
    if i==0:
         dfl2_logDiff.append(0)
    else:
         dfl2_logDiff.append(np.log(i) - np.log(df2['Close'][i-1]))

df3 = web.DataReader('SPY', 'yahoo', start, end)
df3 = df3[['Open', 'High', 'Low', 'Close']]
df3.reset_index(inplace=True)
df3['Date'] = df3['Date'].map(mdates.date2num)
dft3_earnings = []
for i in df3['Close']: 
    dft3_earnings.append(i / df3['Close'][0])
dfl3_logDiff = []
for i in df3['Close']: 
    if i==0:
         dfl3_logDiff.append(0)
    else:
         dfl3_logDiff.append(np.log(np.round(i)) - np.log(np.round(df3['Close'][i-1])))

ax = plt.subplot()

#Original Price
#plt.plot(df['Date'], df2['Close'], label="AAPL")
#plt.plot(df1['Date'], df2['Close'], label="GOOG")
#plt.plot(df2['Date'], df2['Close'], label="MSFT")

#Returns
#plt.plot(df['Date'], dft0_earnings, label="AAPL")
#plt.plot(df1['Date'], dft1_earnings, label="GOOG")
#plt.plot(df2['Date'], dft2_earnings, label="MSFT")
#plt.plot(df3['Date'], dft3_earnings, label="SPY")

#Log Price difference
plt.plot(df['Date'], dft0_logDiff, label="AAPL")
plt.plot(df1['Date'], dft1_logDiff, label="GOOG")
plt.plot(df2['Date'], dft2_logDiff, label="MSFT")
plt.plot(df3['Date'], dft3_logDiff, label="SPY")

plt.legend()
ax.grid(True)
ax.set_axisbelow(True)
ax.set_title('Share Price', color='white')
ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.xaxis_date()
plt.show()
