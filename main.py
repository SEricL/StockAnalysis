import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as plt
import yfinance as yf
import bs4
import requests

# Your chosen stock
ticker = 'AAPL'

tickerData = yf.Ticker(ticker)

# Time frame for stock data.
tickerDf = tickerData.history(period='1d', start='2013-1-1', end='2023-5-1')

# Drop missing rows, make target data frame
tickerDf.dropna()

# Drop information that's inaccessible till after price is determined
tickerDf.drop('High', axis=1, inplace=True)
tickerDf.drop('Low', axis=1, inplace=True)
tickerDf.drop('Volume', axis=1, inplace=True)  # Volume might be worth keeping

y = tickerDf['Close']  # Close is the result we want to find, make a result set
tickerDf.drop('Close', axis=1, inplace=True)

# Default test size is 30%, training set is 70%
trainX, testX, trainY, testY = train_test_split(tickerDf, y, test_size=0.3, random_state=42)

# Make and train a linear regression model
linear = linear_model.LinearRegression()
linear.fit(trainX, trainY)

# Get very recent data and test model
testFrame = tickerData.history(period='1d', start='2023-5-7', end='2023-5-11')
testFrame.drop(columns=['High', 'Low', 'Volume', 'Close'], inplace=True)

# Test model's accuracy on its own test set(old data). Accuracy is extremely high bcs bias
# since it's overtuned when using daily data
# prediction = linear.predict(testX)
# print("Accuracy:", linear.score(testX, testY))

# Print prediction on recent data
pred = linear.predict(testFrame)
print(pred)

single = testFrame.iloc[:]['Open'][-1]
singleClose = pred[-1]

# Difference in price
difference = singleClose - single
# Percent difference
percentage = difference/single*100

sia = SentimentIntensityAnalyzer()
vaderCoeff = 1  # General weight for vader sentiment analysis, use/modify if sentiment analysis weight isn't important


# More research required to figure out a better weighing scheme. Current one is prone to errors, and not very good
# [News headline, article importance]
headlines = [["Apple reports a drop in income", 0.9]]
outputs = []

# Converts headlines to polarity scores, keeps article importance/weights
for a in headlines:
    outputs.append([sia.polarity_scores(a[0]), a[1]])

totalNews = 0

# Sum weights of articles * their compound value
for each in outputs:
    if each[0]['compound'] > 0.05:
        #positive news
        totalNews += each[1]
    elif each[0]['compound'] < 0.05:
        # Negative news
        totalNews -= each[1]
temp = totalNews
totalNews = 1 + (totalNews/len(outputs))

# New predictions
npTotal = singleClose * totalNews
npDifference = difference * totalNews

