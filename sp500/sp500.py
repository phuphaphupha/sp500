#note: handbopok in machine learning to predict sp500 based on previous starting price and closing price using randomforest method 
#we only focus on factors which is ['Close_Ratio_2', 'Trend_2', 'Close_Ratio_5', 'Trend_5', 'Close_Ratio_60', 'Trend_60', 'Close_Ratio_250', 'Trend_250', 'Close_Ratio_1000', 'Trend_1000']
# this is the ratio and trend whether the price goes up or down.

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


#download price history from a single simbol
# if you print out sp500 here, you will get data up to date which can use to analyse
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history (period="max")
#print(sp500)

#index of sp500 is the date
sp500.index

#__________________________________
#cleaning and visualizing stock market

#plot data
#sp500.plot.line(y="Close", use_index = True)
#plt.show()

#remove data that we dont need (remove collumn)
del sp500["Dividends"]
del sp500["Stock Splits"]

#____________________________________
#predict whether the price of tomorrow stocks go up or down

#create another column called tmr price, which is the shifted in date (index) of 1 day
sp500["Tomorrow"] = sp500["Close"].shift(-1)
#create another coulmn (target column) that tells whether the stock goes up or not
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

#remove all data before 1990
sp500 = sp500.loc["1990-01-01":].copy()

#____________________________________
#model training and prediction
#n_estimators are number of branch of the radom forest
#random_state=1 will make the machine asking same question in each brunch
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
#split the data from first row to before last 100 rows for train set
train = sp500.iloc[:-100]
#split the kast 100 rows for the test set
test = sp500.iloc[-100:]

#fit the data
predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

'''

#see the prediction, this preds is numpy array
preds = model.predict(test[predictors])
#change numpy array into series so it is easier to see
preds = pd.Series(preds, index=test.index)
#calculate prediction score
score = precision_score(test["Target"], preds)
print(score)
#plot to show between the actual value and predicted value with plot.show
combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()
#plt.show()

'''

#_____________________________
#building a more robust way to test algorithm, backtesting

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    #return probability that stock price will go down or go up, [:1] means go up
    preds = model.predict_proba(test[predictors])[:,1]
    #reduce number of total trading days, if more than 60 percent the price goes up, we need to buy
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0

    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)

    return combined

#start = number of data train for forst round, step = additional data to include for training in the next round
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    #(start = number of data at start, data.shape = total data avilable, step size: e.g. from 200 to 400, step = 50)
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    #cancat take a list and turn into a single data frame
    return pd.concat(all_predictions)


'''
#_______________________ run the test _______________________

#use function above
predictions = backtest(sp500, model, predictors)

#see how many times sp500 goes down and goes up
print(predictions["Predictions"].value_counts())

#see the precision_score
print(precision_score(predictions["Target"], predictions["Predictions"]))
print(predictions)

'''

#add more predictor to model
#number of days (previous) to take average
horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    #take the average
    rolling_averages = sp500.rolling(horizon).mean()

    #create new column with ratio for each horizon
    ratio_column = f"Close_Ratio_{horizon}"

    #ratio of today close and the average in the previous horizon days
    sp500[ratio_column] = sp500["Close"]/rolling_averages["Close"]

    #trend column to see whether each period goes up or not
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    #add into new predictor
    new_predictors += [ratio_column, trend_column]

#print(sp500) (print to see hows it going)

print(new_predictors)

#drop row with nan (value that cannot be calculated)
sp500 = sp500.dropna()

predictions = backtest(sp500, model, new_predictors)
valuecount = predictions["Predictions"].value_counts()
print(valuecount)

#precision score for the final prediction
print(precision_score(predictions["Target"], predictions["Predictions"]))
