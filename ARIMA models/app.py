# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Building ARIMA Models
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from keras.models import load_model
import streamlit as st


# START
st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AMZN')
path = '../datasets/' + user_input + '.csv'
df = pd.read_csv(path, header=0, index_col=0)

# Convert index to datetime64
df.index = pd.to_datetime(df.index)
start = df.index.min()
end = df.index.max()
print('Start date: ', start)
print('End date: ', end)

# Describing Data
start = df.min()
st.subheader(f'Data from {start.year} - {end.year}')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

# Plot with moving average
st.subheader('Closing Price vs Time chart with Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,4))
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

# Split the data for train and test
df_train = df.Close[:1228].copy()
df_test = df.Close[1228:].copy()

# Creating function to display ACF anf PACF plot
def plot_correlation(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    # ACF
    plot_acf(df, ax=ax1, lags=30)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    
    # PACF
    plot_pacf(df, ax=ax2, lags=20)
    plt.xlabel('Lag')
    plt.ylabel('Partial Autocorrelation')

    # Lighten the borders
    ax1.spines['top'].set_alpha(.3); ax2.spines['top'].set_alpha(.3)
    ax1.spines['bottom'].set_alpha(.3); ax2.spines['bottom'].set_alpha(.3)
    ax1.spines['right'].set_alpha(.3); ax2.spines['right'].set_alpha(.3)
    ax1.spines['left'].set_alpha(.3); ax2.spines['left'].set_alpha(.3)

    ax1.tick_params(axis='both', labelsize=10)
    ax2.tick_params(axis='both', labelsize=10)
    st.pyplot(fig)

plot_correlation(df_train)

#perform augmented Dickey-Fuller test to check if stationary
def ADF(df):
    result = adfuller(df)

    # Extract ADF Values
    print('Column Name: %s' % "Close Variable")
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

ADF(df_train)

def to_stationary(df):
    df_diff = df.diff().dropna()
    if df_diff[1] <= 0.05:
        print(f"p.value: {df_diff[1]}")
        return df_diff
    else:
        return to_stationary(df_diff)

df_train_diff = to_stationary(df_train)

# Plotting Close Trend after differencing
st.subheader('Close Trend After Differencing')
fig = plt.figure(figsize=(12,4))
plt.plot(df_train_diff)
print(f"p.value: {df_train_diff[1]}")
st.pyplot(fig)

# library that determine best parameters for ARIMA model
import pmdarima as pm
from pmdarima import auto_arima

best_model = auto_arima(df_train, start_p=0, start_q=0,
                          max_p=3, max_q=3,m=12,start_P=0,start_Q=0, 
                          max_P = 3, max_Q = 3,
                          seasonal=True,
                          d=1,D=1,trace=True,
                          error_action='ignore',   
                          suppress_warnings=True,  
                          stepwise=True)

# Get best SARIMA model
order = best_model.order
seasonal_order = best_model.seasonal_order

# Splitting data into training and testing
training = pd.DataFrame(df.Close[0:int(len(df.Close)*0.95)])
testing = pd.DataFrame(df.Close[int(len(df.Close)*0.95):int(len(df.Close))])

# Fitting the model
final_model = SARIMAX(training,order=order,seasonal_order=seasonal_order)
result = final_model.fit()
print(result.summary())

# Plot model result
st.subheader('Model Analysis')
fig = result.plot_diagnostics(figsize=(16, 12))
st.pyplot(fig)


# Obtain predicted values
start=len(training)
end=len(training)+len(testing)-1
predictions = result.predict(start=start, end=end, dynamic=False, typ='levels').rename(f'SARIMA{order}{seasonal_order} Predictions')# Plot predictions against known values

# Join index
predictions.index = testing.index
clean_df = pd.concat([predictions, testing], axis=1)
clean_df

# Plotting the prediction vs testing
st.subheader('Predicted vs Original')
fig = plt.figure(figsize=(12,6))
plt.plot(testing, 'b', label = 'Original Price')
plt.plot(predictions, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


