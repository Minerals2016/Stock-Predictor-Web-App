#replacement for current API: https://rapidapi.com/sparior/api/mboum-finance

#Library import 
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#other
#from datetime import datetime
#from pandas_datareader import data
import yfinance as yf #yahoo finance API
from datetime import date #get current date

#get stock price 
#df = web.DataReader('TSLA', data_source='nasdaq', start='2012-01-01', end='2019-12-17')
#show data
#df


"""
tickers = 'AAPL'
start_date = '1980-12-01'
end_date = '2018-12-31'
stock_data = data.get_data_yahoo(tickers, start_date, end_date)
"""


"""
stock = 'AAPL'
ls_key = 'Adj Close'
start = datetime(2012,1,1)
end = datetime(2019,12,17)    
df = web.DataReader(stock, 'google',start,end)
df
"""



#get today's date
today = date.today()

#get stock information
symbol = "AAPL"
stock = yf.Ticker(symbol)
df = stock.history(interval='1d', start='2012-01-01', end='2019-12-17')
#df

#visualize the closing price history
plt.figure(figsize=(16, 8))
plt.title("$" + symbol + " Close Price History")
plt.plot(df["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price in USD ($)", fontsize=18)
#plt.show

#Create new dataframe with only the close column
data = df.filter(["Close"])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)

training_data_len
#len(data.values)

#Scale the data 
scaler = MinMaxScaler(feature_range=(0,1)) 
scaled_data = scaler.fit_transform(dataset) #compute for scaling from 0-1

scaled_data

#Create training dataset
#Create the scaled training dataset
train_data=scaled_data[0:training_data_len, :]

#split the data into x_train and y_train data sets
x_train = [] #independent training features
y_train = [] #tarfet variables 

for i in range(60, len(train_data)):     #question: what does the 60 do? No output when training_data_len = 54 
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<= 61:
    print(x_train)
    print(y_train)
    print(" ")

#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#video timestamp 21:36
#https://www.youtube.com/watch?v=QIUxPv5PJOY

#Build the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Create testing dataset
#Create a new array containing scaled values from index 397 to 571
test_data = scaled_data[training_data_len - 60: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i,0])

#Convert the data to a numpy array
x_test = np.asarray(x_test)

#Reshape the data
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean( predictions - y_test)**2)
rmse

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title("$" + symbol + " Prediction Model")
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
#plt.show()

#Worked till video timestamp 40:30
#https://youtu.be/QIUxPv5PJOY?t=2430

#Show the valid and predicted prices 
#valid

#Get the quote

#get today's date
#today = date.today()

#get stock information
symbol2 = "AAPL"
stock2 = yf.Ticker(symbol2)
df2 = stock2.history(interval='1d', start='2012-01-01', end='2019-12-17')

#Create new dataframe with only the close column
new_df = df2.filter(["Close"])

#get the last 60 day closing price and convert the dataframe to an array
last_60_days = new_df[-60:].values

#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#create empty list 
X_test = []

#Append the past 60 days

X_test.append(last_60_days_scaled)

#Convert the X_test data set to a numpy array 
X_test = np.array(X_test)

#Reshape the data 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Get the predicted scaled price
pred_price = model.predict(X_test)

#undo the scaling
pred_price = scaler.inverse_transform(pred_price)

#print prediction
print(pred_price)

#Get the actual quote
symbol3 = "AAPL"
stock3 = yf.Ticker(symbol3)
df3 = stock2.history(interval='1d', start='2019-12-18', end='2019-12-19')
print(df3["Close"].values)

import pickle
with open('stock_predictor_model_a.pkl', 'wb') as file:
	pickle.dump(model, file)