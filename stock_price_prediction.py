import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('DataSet/StockPricePredDataSet.csv', date_parser=True)
train_set = dataset[dataset['Date'] < '2019-01-01']
test_set = dataset[dataset['Date'] > '2019-01-01']

train_set = train_set.drop(['Date', 'Adj Close'], axis=1)

scaler = MinMaxScaler()
train_set = scaler.fit_transform(train_set)

# prepare train dataset
X_train = []
y_train = []

for i in range(60, train_set.shape[0]):
    X_train.append(train_set[i - 60:i])
    y_train.append([i, 0])

# convert list to array
X_train = np.array(X_train)
y_train = np.array(y_train)

# Build LSTM Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

regressor = Sequential()
regressor.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape, 5)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=70, activation='relu', return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=90, activation='relu', return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=110, activation='relu'))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1))

# get model summary
regressor.summary()

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs=60, batch_size=64)

# prepare test dataset
last_60_days_from_train_set = train_set.tail(60)
test_set = last_60_days_from_train_set.append(test_set, ignore_index=True)
test_set = test_set.drop(['Date', 'Adj Close'], axis=1)
test_set = scaler.fit_transform(test_set)

X_test = []
y_test = []

# prepare test dataset
for i in range(60, test_set.shape[0]):
    X_test.append(test_set[i - 60:i])
    y_test.append([i, 0])

# convert list to array
X_test = np.array(X_test)
y_test = np.array(y_test)

# predict values
y_pred = regressor.predict(X_test)

# inverse of MinMaxScaler
# 1. get the scaler
scale = 1 / scaler.scale_[0]
# 2.get real y value
y_pred = y_pred * scale
y_test = y_test * scale

# Visualizing the output
