# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Building ARIMA Models
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# from keras.models import load_model
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

# Describing Data
st.write(f'Data from {start.year} - {end.year}')
st.write(df.describe())

# Select close column for fitting the model
df_close = df.Close
df_close.info()

# Observe Close dataset
st.subheader(f'Determine Annual Trend and Seasonality of {user_input}')
fig = plt.figure(figsize=(16,5))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_close, 'green', label='Close Trend')
plt.legend()
st.pyplot(fig)


# Plot with moving average
st.subheader('Closing Price vs Time chart with Moving Average')
ma100 = df_close.rolling(100).mean()
ma200 = df_close.rolling(200).mean()
fig = plt.figure(figsize=(12,4))
plt.plot(df_close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)


# Exploratory Analysis / Preprocessing
st.subheader('Exploratory Analysis / Preprocessing')
st.write(df_close.describe())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
df_close.plot(title='Histogram', kind='hist', ax=ax1)
ax1.set_axisbelow(True)
ax1.grid(color='gray', linestyle='dashed')
df_close.plot(title='Density', kind='kde', ax=ax2)
ax2.grid(color='gray', linestyle='dashed')
st.pyplot(fig)

st.write('Data Skewness: ', df_close.skew())


# Functions to determine ARIMA parameter p,d,q
# st.write('Creating function to display ACF anf PACF plot for Close Dataframe')
def plot_correlation(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
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

st.write('Perform augmented Dickey-Fuller test to check if stationary for Close Dataframe')
def ADF(df):
    result = adfuller(df)

    # Extract ADF Values
    print('Column Name: %s' % "Close Variable")
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    st.write('P-value: ', result[1])

st.write('Determining Differencing value for Close Dataframe')
# Perform differencing
def to_stationary(df):
    cycle = 0
    df_diff = df.diff().dropna()
    cycle += 1
    if df_diff[1] <= 0.05:
        print(f"p.value after differencing: {df_diff[1]}")
        st.write(f"Cycle of differencing: {cycle}")
        return df_diff
    else:
        return to_stationary(df_diff)


# Plotting Close Trend after differencing
def plot_dif(df):
    df_diff = to_stationary(df)
    st.write('Close Trend After Differencing')
    fig = plt.figure(figsize=(12,4))
    plt.plot(df_diff)
    st.pyplot(fig)
    st.write(f"p.value after differencing: {df_diff[1]}")


# Split the data for train and test
row_len = int(len(df_close)*0.9)
df_train = list(df_close[0:row_len].copy())
df_test = list(df_close[row_len:].copy())

# Plot training and testing data
st.subheader('Plotting Train and Test Split')
fig = plt.figure(figsize=(16,5))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_close[0:row_len], 'green', label='Train data')
plt.plot(df_close[row_len:], 'blue', label='Test data')
plt.legend()
st.pyplot(fig)


# Getting ARIMA parameters for training dataset
st.subheader('Determining ARIMA parameter p,d,q')
training = df_close[0:row_len]
# Data Visualization after differencing
plot_dif(training)
# ACF and PACF
st.write('Autocorrelation and Partial Autocorrelation')
plot_correlation(training)
# AFT
st.write('Perform Augmented Fuller test')
ADF(training)

# ARIMA fitting
st.subheader('Fit the ARIMA Model')
st.write('Get order value for ARIMA model')
p = st.number_input('Enter Autoregressive (p) value: ', step=1)
d = st.number_input('Enter Differencing (d) value: ', step=1)
q = st.number_input('Enter Moving Average (q) value: ', step=1)
ARIMA_order = (p,d,q)
st.write(ARIMA_order)

# Create an array of predicted value
model_predictions = []
n_test_observ = len(df_test)

# Iteration to get predicted value to each trained model
for i in range(n_test_observ):
    model = ARIMA(df_train, order=ARIMA_order)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    actual_test_value = df_test[i]
    df_train.append(actual_test_value)

st.write(model_fit.summary())

# Plot Predicted VS Actual Price
st.subheader('Predicted VS Actual Price')
fig = plt.figure(figsize=(12,5))
plt.grid(True)
date_range = df_close[row_len:].index
plt.plot(date_range, model_predictions, color = 'blue', linestyle = 'dashed', label = 'Predicted Price')
plt.plot(date_range, df_test, color = 'green', label = 'Actual Price')

plt.title(f'{user_input} Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Performance Evaluation using MAPE
st.subheader('Model performance evaluation')
mape = np.mean(np.abs(np.array(model_predictions) - np.array(df_test))/np.abs(df_test))
st.write('MAPE: ', mape) # Mean Absolute Percentage Error
