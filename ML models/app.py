# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st

# load dataset from Computer
# df_AMZN = pd.read_csv('../datasets/AMZN.csv', header=0, index_col=0)
# df_AMZN.head()

# Web scrape dataset
import pandas_datareader as data
start = '2019-01-01'
end = '2023-01-01'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AMZN')
path = '../datasets/' + user_input + '.csv'
# df = data.DataReader(user_input, 'yahoo', start, end)
df = pd.read_csv(path, header=0, index_col=0)

# Describing Data
st.subheader('Data from 2019 - 2023')
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
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)


# Splitting data into training and testing
training = pd.DataFrame(df.Close[0:int(len(df.Close)*0.70)])
testing = pd.DataFrame(df.Close[int(len(df.Close)*0.70):int(len(df.Close))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

training_array = scaler.fit_transform(training)

# Splitting data into X train and Y train

# Load model
model = load_model('keras_model.h5')

# Testing part
past_100days = training.tail(100)
final_df = past_100days.append(testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions
y_prediction = model.predict(x_test)

# Scale down
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_prediction = y_prediction * scale_factor
y_test = y_test * scale_factor

# Final Visualization
st.subheader('Predicted vs Original')
fig = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_prediction, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)