import pdb
import mymodule
#Stocker Library Code Example
from stocker import Stocker

microsoft = Stocker('MSFT')

stock_history = microsoft.stock
stock_history.head()
microsoft.plot_stock()

microsoft.plot_stock(start_date = '2000-01-03', end_date = '2018-01-16', stats ['Daily Change', 'Adj.Volume'], plot_type='pct')

microsoft.buy_and_hold(start_date = '1986-03-13', end_date='2018-01-16', nshares=100)

model, model_data = microsoft.create_prophet_model()
model.plot_components(model_data)
plt.show()

print(micorsoft.weekly_seasonality)
microsoft.weekly_seasonality = True
print(microsoft.weekly_seasonality)

#only shows data in the first 80% of data
microsoft.changepoint_date_analysis()

#This will need to add a library of search terms related to the stocks information
microsoft.changepoint_date_analysis(search = 'Microsoft profit')
microsoft.changepoint_date_analysis(search = 'Microsoft Office')

#Future predictions
model, future = microsoft.create_prophet_model(days=180)

microsoft.evaluate_prediction()
