import quandl
import pandas as pd
import numpy as np

mydata = quandl.get("YAHOO/INDEX_DJI", start_date="2005-12-01", end_date="2005-12-05")
authtoken ='XXX'

def get_data_quandl(symbol, start_date, end_date):
    data = quandl.gty(symbol, start_date=start_date, end_date=end_date, authtoken=authtoken)
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
    df_new['avg_price_5'] = pd.rolling_mean(df['Close'], window=5).shift(1)
    #rolling_mean calculates the moving average given a window
    df_new['avg_price_30'] = pd.rolling_mean(df['Close'], window=21).shift(1)
    df_new['avg_price_365'] = pd.rolling_mean(df['Close'], window=252).shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
    #average volume
    df_new['avg_volume_5'] = pd.rolling_mean(df['Volume'], window=5).shift(1)
    df_new['avg_volume_30'] = pd.rolling_mean(df['Volume'], window=21).shift(1)
    df_new['avg_volume_365'] = pd.rolling_mean(df['Volume'], window=252).shift(1)
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
    weights_delta = np.dot(X_train.T, y_train - predictions)
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
    predictions = compute_prediction(X, weights)
    cost = np.mean((predictions - y) **2 /2.0)
    return cost

def train_linear_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False):
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
    weights = np.zero(X_train.shape[0], 1)
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, y_train, weights, learning_rate)
        #Cheack the cost for everry 100 (fopr exampe) iterations
        if iteration % 100 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights

def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    return compute_prediction(X, weights)

#Example set of data
X_train = np.array([[6], [2], [3], [4], [1], [5], [2], [6], [4], [7]])
y_train = np.array([5.5,1.6, 2.2, 3.7, 0.8, 5.2, 1.5, 5.3, 4.4, 6.8])
weights = train_linear_regression(X_train, y_train, max_iter=100, learning_rate=0.01, fit_intercept=True)

X_test = np.array([[2.3], [3.5], [5.2], [2.8]])
predictions = predict(X_test, weights)
import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], y_train, markker='o', c='b')
plt.scatter(X_test[:, 0], predictions, marker='*', c='k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

from sklearn.linear_model import SGDRegressor
regressor = SGDRegressor(loss='squared_loss', penalty='12', alpha=0.0001, learning_rate='constant', eta=0.01, n_iter=1000)

#Decision tree regression
def mse (targets):
    #When the set is empty
    if targets.size == 0:
        return 0
    return np.var(targets)

def weighted_mse(groups):
    """Calculate weighted MSE of children after a split
    Args:
        groups (list of children, and a child consists a list of targets)
    Returns:
        float, weighted impurity
        """
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * mse(group)
        return weighted_sum

print('{0:.4f}'.format(mse(np.array([1,2,3]))))

def split_node(X, y, index, value):
    """Split data set X,y basedon a feature and a value
    Args:
        X, y (numpy.ndarray, data set)
        index (int, index of the feature used for splitting)
        value (value of the feature used for splitting)
    Returns:
        list, list: left and right chilf a chilf s in the format of [X,y]
        """
    x_index = X[:, index]
    #if this feature is numerical
    if type(X[0, index]) in [int, float]:
        mash = x_index >=value
        #if this feature is categorical
    else:
        mask = x_index== value
    #split into left and right child
    left = [X[~mask, :], y[~mask]]
    right = [X[mask, :], y[mask]]
    return left, right

def get_best_split(X, y):
    """Obtain the best splitting point and resulting children
    for the data set X, y
    Args:
        X, y (numpy.ndarray, data set)
        criterion (gini or entropy)
    Returns:
        dict {indec: index pf the feature, value: feature value, children: ;eft and right children}
        """
    best_index, best_value, best_score, children = None, None, 1e10, None
    for index in range(len(X[:, best_index])):
        groups = split_node(X, y, index, best_value)
        impurity = weighted_mse([groups[0][1], groups[1][1]])
        if impurity < best_score:
            best_index, best_value, best_score, children = index, best_value, impurity, groups
    return {'index': best_index, 'value': best_value, 'children': children}

def get_leaf(targets):
    #obtain the leaf as the mean of the targets
    return np.mean(targets)

def split(node, max_depth, min_size, depth):
    """Split children of a node to construct new nodes opr assign them terminals
    Args:
        node(dict, with children info)
        max_depth (int, maximal deepth of the tree)
        min_size (int, minimal samples required to further split a child)
        depth (int, current depth of the node)
        """
    left, right = node['children']
    del (node['children'])
    if left[1].size ==0:
        node['right'] = get_leaf(right[1])
        return
    if right[1].size ==0:
        node['left'] = get_leaf(left[1])
        return
    #Check if the current depth exceeds the maximal depth
    if depth >= max_depth:     
        node['left'], node['right'] = get_leaf(left[1]), get_leaf(right[1])
        return
    #Check if the left child has enough samples
    if left[1].size <= min_size:
        node['left'] = get_leaf(left[1])
    else:
        #It has enough samples, we further split it 
        result = get_best_split(left[0], left[1])
        result_left, result_right = result['children']
        if result_left[1].size ==0:
            node['left'] = get_leaf(result['children'])
        elif result_right[1].size ==0:
            node['left'] = get_leaf(result_left[1])
        else:
            node['left'] = result
            split(node['left'], max_depth, min_size, depth + 1)

    #Check if the right child has enough samples
    if right[1].size <= min_size:
        node['right'] = get_leaf(right[1])
    else:
        #Ithas enough samples, we further split it
        result = get_best_split(right[0], right[1])
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['right'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['right'] = get_leaf(result_left[1])
        else:
            node['right'] = result
            split(node['right'], max_depth, min_size, depth + 1)

def train_tree(X_train, y_train, max_depth, min_size):
    """COnstruction of a tree starts here
    Args:
        X_train, y_train (list, list, training data)
        max_depth (int, maximal depoth of the tree)
        min_size (int, minimal samples required to further split a child)
        """
    root = get_best_split(X_train, y_train)
    split(root, max_depth, min_size, 1)
    return root

X_train = np.array([['semi', 3], ['detached', 2], ['detached',3], ['semi', 2], ['semi', 4]], dtype=object)
y_train = np.array([600, 700, 800, 400, 700])
tree = train_tree(X_train, y_train, 2, 2)

CONDITION = {'numerical': {'yes': '>=', 'no': '<'}, 'catagorical': {'yes': 'is', 'no': 'is not'}}

def visualize_tree(node, depth=0):
    if isinstance(node, dict):
        if type(node['value']) in [int, float]:
            condition = CONDITION['numerical']
        else:
            condition = CONDITION['CATAGORICAL']
       
        print('{}|-X{} {} {}'.format(depth* '  ', node['index'] + 1, condition['no'], node['value']))
        if 'left' in node:
            visualize_tree(node['left'], depth +1)
        print('{}|-X{} {} {}'.format(depth* '  ', node['index'] + 1, condition['yes'], node['value']))
        if 'right' in node:
            visualize_tree(node['right'], depth +1)
    else:
        print('{}[{}]'.format(depth * '  ', node))

visualize_tree(tree)

boston = datasets.load_boston()
num_test = 10 #the last 10 samples as testing set
X_train = boston.data[:-num_test, :]
y_train = boston.target[:-num_test:]
X_test = boston.data[-num_test:, :]
y_test = boston.target[-num_test:]
from sklearn.tree import DecisionTreeRegressor
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions)

print(y_test)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=3)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions)

#Support Vector Regression
from sklearn.svm import SVR
regressor = SVR(C=0.1, epsilon=0.02, kernel='linear')
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions)

#Regression performace evaluatio = dataset.;pad_diabetes()
num_test = 30 #the last 30 samples as testing set
X_train = diabetes.data[:-num_test, :]
y_train = diabetes.target[:-num_test]
X_test = diabetes.data[-num_test:, :]
y_test = diabetes.target[-num_test:]
param_grid = {"alpha": [1e-07, 1e-06, 1e-05], "penalty":[None, "12"], "eta0":[0.001, 0.005, 0.01], "n_iter": [300, 1000, 3000]}
from sklearn.model_selection import GridSearchCV
regressor = SGDRegressor(loss='squared_loss', learning_rate='constant')
grid_search = GridSearchCV(regressor, param_grid, cv=3)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
#('penalty':None, 'alpha': 1e-05, 'eta':0.01, 'n_iter': 300}
regressor_best = grid_search.best_estimator_
predictions = regressor_best.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
mean_squared_error(y_test, predictions)
mean_absolute_error(y_test, predictions)
r2_score(y_test, predictions)

#Predictions
import datetime
start_train = datetime.datetime(1988, 1, 1, 0, 0)
end_train = datetime.datetime(2014, 12, 31, 0, 0)
data_train = data.ix[start_train:end_train]

X_columns = list(data.drop(['close'], axis=1).columns)
y_column = 'close'
x_train = data_train[X_columns]
y_train = data_train[y_column]

x_train.shape
y_train.shape

start_test = datetime.datetime(2015, 1, 1, 0, 0)
end_test = datetime.datetime(2015, 12, 31, 0, 0)
data_test =data.ix[start_test:end_test]
X_test = data_test[X_columns]
y_test = data_test[y_column]

X_test.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transfrom(X_train)
X_scaled_test = scaler.transform(X_test)

param_grid = {
        "alpha": [3e-06, 1e-05, 3e-5],
        "eta0": [0.01, 0.03,0.1],}
lr =SGDRegressor(penalty='12', n_iter=1000)
grid_search = GridSearchCV(lr,param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_scaled_train, y_train)

print('MSE:{0:.3f}'.format(mean_squared_error(y_test, predictions)))
print('MAE: {0:.3f}'.format(mean_absolute_error(y_test, predictions)))
print('R^2: {0:.3f}'.format(r2_score(y_test, predictions)))

param_grid = {
        "max_depth": [30,50],
        "min_samples_split": [3,5,10],
        }
rf = RandomForestRegressor(n_estimators=1000)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
rf_best = grid_search.best_estimator_
predictions = rf_best.predict(X_test)

print('MSE: {0:.3f}'.format(mean_squared_error(y_test, predictions)))
print('MSE: {0:.3f}'.format(mean_absolute_error(y_test, predictions)))
print('R^2: {0:.3F}'.format(r2_score(y_test, predictions)))


#SVR
param_grid = {
        "C": [1000, 3000,10000],
        "epsilon": [0.00001, 0.0003, 0.0001],
        }
svr = SVR(kernel='linear')
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_scaled_train, y_train)
print(grid_search.best_params_)

svr_best = grid_search.best_estimator_
predictions = svr_best.predict(X_scaled_test)
print('MSE: {0:.3f}'.format(mean_squared_error(y_test, predictions)))
print('MSE: {0:.3f}'.format(mean_absolute_error(y_test, predictions)))
print('R^2: {0:.3F}'.format(r2_score(y_test, predictions)))






