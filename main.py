import nltk
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
y = tickerDf['Close']
tickerDf.drop('Close', axis=1, inplace=True)

# Default test size is 30%, training set is 70%
train, trainY, test, testY = train_test_split(tickerDf, y, test_size=0.3, random_state=42)
linear = linear_model.LinearRegression()
linear.fit(train, trainY)
