import nltk
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib as plt
import yfinance as yf

# Your chosen stock
ticker = 'AAPL'

tickerData = yf.Ticker(ticker)

# Time frame for stock data.
tickerDf = tickerData.history(period='1d', start='2013-1-1', end='2023-5-1')

# Drop missing rows, make target data frame
tickerDf.dropna()
tickerDf.drop('High', axis=1, inplace=True)
tickerDf.drop('Low', axis=1, inplace=True)
tickerDf.drop('Volume', axis=1, inplace=True)
y = tickerDf['Close']
tickerDf.drop('Close', axis=1, inplace=True)

# Default test size is 30%, training set is 70%
trainX, testX, trainY, testY = train_test_split(tickerDf, y, test_size=0.3, random_state=42)
linear = linear_model.LinearRegression()
linear.fit(trainX, trainY)

testFrame = tickerData.history(period='1d', start='2023-5-7', end='2023-5-11')
testFrame.drop(columns=['High','Low', 'Volume','Close'], inplace=True)

#prediction = linear.predict(testX)
# print("Accuracy:", linear.score(testX, testY))
print(linear.predict(testFrame))