import pandas as pd
import numpy as np 

from datetime import datetime

import seaborn as sns
import cufflinks as cf
import plotly.offline as py
import matplotlib.pyplot as plt

from pylab import rcParams
import statsmodels.api as sm

from pmdarima.arima import auto_arima

import pickle

# Load datasets
url_train_a='https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv'
url_train_b='https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv'
url_test_a='https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv'
url_test_b='https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv'

train_a=pd.read_csv(url_train_a)
train_b=pd.read_csv(url_train_b)
test_a=pd.read_csv(url_test_a)
test_b=pd.read_csv(url_test_b)

# change 'datetime' type to 'datetime64'
train_a['datetime']=train_a['datetime'].astype('datetime64')
train_b['datetime']=train_b['datetime'].astype('datetime64')
test_a['datetime']=test_a['datetime'].astype('datetime64')
test_b['datetime']=test_b['datetime'].astype('datetime64')

# Set index
train_a.set_index('datetime',inplace=True)
test_a.set_index('datetime',inplace=True)
train_b.set_index('datetime',inplace=True)
test_b.set_index('datetime',inplace=True)

# Seasonal ARIMA Model
# Define the model
stepwise_model = auto_arima(train_a, start_p=1, start_q=1,
                           max_p=1, max_q=1, m=60*2,
                           start_P=0, seasonal=True,
                           d=1, D=1,max_P=1,max_Q=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True,random_state=608)
print(stepwise_model.aic())

# Fit the best model
stepwise_model.fit(train_a)

# Predictions
future_forecast_a = stepwise_model.predict(n_periods=60)

# Save the model as a pickle
filename = '/workspace/alt-time-series/modelss/best_model_a.pkl'
pickle.dump(stepwise_model, open(filename,'wb'))

# Reffiting the model
stepwise_model.fit(train_b)
future_forecast_b = stepwise_model.predict(n_periods=60)

# Save the model as a pickle
filename = '/workspace/alt-time-series/models/best_model_b.pkl'
pickle.dump(stepwise_model, open(filename,'wb'))
