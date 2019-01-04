import quandl

myudata = quandl.get("YAHOO/INDEX_DJI", start_date="2005-12-01", end_date="2005-12-05")
mydata

authtoken ='XXX'

def get_data_quandl(symbol, start_date, end_date):
  data = quandl.gty(symbol, start_date=start_date, end_date, authtoken=authtoken)
  return data

def generate_features(df):
  """Generate features for a stock/index based on historical price and performaance
  Args:
  df (dataframe with columns "Open", "Close", "High", "Low', "Volume", "Adjusted Close")
  Returns:
  dataframe, data set with new features
  """
  df_new=pd.DataFrame()
  #6 original features
  df_new['open'] = df['Open']
  df_new['open_1'] = df['Open'].shift(1)
  #shift index by 1, in order to take the value of the previos day.
  df_new['close_1'] = df['Close'].shift(1)
  df_new['high_1'] = df['High'].shift(1)
  df_new['low_1'] = df['Low'].shift(!)
  df_new['volume_1'] = df['Volume'].shift(1)
  #31 orginial features
  #average prive
  df_new['avg_price_5'] = pd.rolling_mean(df['Close'], window=5),shift(1)
  #rolling_mean calculates the moving average given a window

