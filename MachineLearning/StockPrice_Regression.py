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
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)
    #31 orginial features
    #average prive
    df_new['avg_price_5'] = pd.rolling_mean(df['Close'], window=5),shift(1)
    #rolling_mean calculates the moving average given a window
    df_new['avg_price_30'] = pd.rolling_mean(df['Close'], window=21).shift(1)
    df_new['avg_price_365'] = pd.rolling_mean(df['Close'], window=252).shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
    #average volume
    df_new['avg_volume_5'] = pd.rolling_mean(df['Volume'], window=5).shift(1)
    df_new['avg_volume_30'] = pd.rolling_mean(df['Volume'], window=21).shift(1)
    df_new['avg_volume_365'] = pd_rolling_mean(df['Volume'], window=252).shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']
    #standard deviation of prices
    df_new['std_price_5'] = pd.rolling_std(df['Close'], window=5).shift(1)
    #rolling_mean calculates the moving standard deviation given a window
    df_new['std_price_30'] = pd.rolling_std(df['Close'], window=21).shift(1)
    df_new['std_price_365'] = pd.rolling_std(df['Close'], window=252).shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']
    #standard deviation of volumes
    df_new['std_volume_5'] = pd.rolling_std(df['Volume'], window=5).shift(1)
    df_new['std_volume_30'] = pd.rolling_std(df['Volume'], window=21).shift(1)
    df_new['std_volume_365'] = pd.rolling_std(df['Volume'], window=252).shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']
    #return
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['CLose'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    #Moving Average
    df_new['moving_avg_5'] = pd.rolling_mean(df_new['return_1'], window=5)
    df_new['moving_avg_30'] = pd.rolling_mean(df_new['return_1'], window=21)
    df_new['moving_avg_365'] = pd.rolling_mean(df_new['return_1'], window=252)
    #the target
    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    #This will drop row with any N/A value, which is by product moving avg/std
    return df_new

symbol = 'YAHOO/INDEX_DJI'
start = '2001-01-01'
end = '2014-12-31'
data_raw = get_data_quandl(symbol, start, end)
data = generate_features(data_raw)

data.round(decimals=3).head(3)


def compute_prediction(X, weights):
    """Compute the predition y_hat based on current weights
    Args:
        X (numpy.ndarray)
        weights (numpy.ndarray)
    Returns:
        numpyl.ndarray, y_hat of X under weights
        """
    predictions = np.dot(X, weights)
    return predictions

def update_weights_gd(X_train, y_tain, weights,learning_rate):
    """Update weights by ine step
    Args:
        X_train, y_train (numpy.ndarray, training data set)
        weights (numpy.ndarray)
        learning_rate(float)
    Return:
        numpy.ndarray, update weights
    """
    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, y_train - preditions)]
    m = y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights

def compute_cost(X, y, weights):
    """Compute the cost j(w)
    Args:
        X, y (numpy.ndarray, data set)
        weights (numpy.ndarray)
    Returns:
        float
    """
    predictions = compute_predictions(X, weights)
    cost = np.mean((predictions - y) **2 /2.0)
    return cost

def train_linear_regression(X_train, y_train, max_iter, learning_ratem fit_intercept=False):
    """Train a linear regression model with graddient descent
    Args:
        X_train, y_train (numpy.ndarray, training data set)
        max_iter (int, number of iterations)
        learning_rate (float)
        fit_intercept (bbol, with an intercept w0 or not)
    Returns:
        numpyt.ndarray,  learned weights
    """
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zero(X_train,shape[0], 1))
    for interation in range(max_iter):
        weights = update_weights_gd(X_train, y_train, weights, learning_rate)
        #Cheack the cost for everry 100 (fopr exampe) iterations
        if iteration % 100 = 0:
            print(compute_cost(X_train, y_train, weights))
    return weigths

def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    return compute_prediction(X, weights)

#Example set of data
X_train = np.array([[6], [2], [3], [4], [1], [5], [2]. [6], [4], [7]])
y_train = np.array([5.5,1.6, 2.2, 3.7, 0.8, 5.2, 1.5, 5.3, 4.4, 6.8])
weights = train_linear_regresion(X_train, y_train, max_iter=100, learning_rate=0.01, fit_intercept=True)

X_test = np.array([[2.3], [3.5], [5 .2], [2.8]])
predictions = predict(X_test, weights)
import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], y_train, markker='o', c='b')
plt.scatter(X_test[:, 0], preditions, marker='*', c='k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

from sklearn.linear_model import SGDREgressor
regressor = SGDRegressor(loss='squared_loss', penalty='12', alpha=0.0001, learning_rate='constant', eta=0.01, n_iter=1000)










