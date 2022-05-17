import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pandas_datareader as data
from keras.models import load_model 
from datetime import date
import streamlit as st
from plotly import graph_objs as go
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

selected_stock = user_input
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

    
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

#describing data
st.subheader('Filtered data')
st.write(df.describe()) 

#visualization
st.subheader('Closing Price vs Time Chart ')
fig = plt.figure(figsize= (12,6))
plt.plot(df.Close)
st.pyplot(fig)



# st.subheader('Closing Price vs Time Chart with 100MA')
# ma100 = df.Close.rolling(100).mean 
# fig = plt.figure(figsize= (12,6))
# plt.plot(ma100)
# plt.plot(df.Close)
# st.pyplot(fig)




# st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
# ma100 = df.Close.rolling(100).mean 
# ma200 = df.Close.rolling(200).mean
# fig = plt.figure(figsize= (12,6))
# plt.plot(ma100)
# plt.plot(ma200)
# plt.plot(df.Close)
# st.pyplot(fig)

#splitting data into testing and trainin
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)


xtrain = []
ytrain = []

for i in range(100,data_training_array.shape[0]):
    xtrain.append(data_training_array[i-100:i])
    ytrain.append(data_training_array[i,0])

xtrain, ytrain = np.array(xtrain), np.array(ytrain)

model = load_model('keras_model.h5')

past_100_days=data_training.tail(100)

final_df=past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])
    
    #making predictions
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

