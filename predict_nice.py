import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def create_data(data):
    x = data[['Close']]
    y = data[['Close']]
    return x, y

def preprocess_data_x(x):
    scaler_x = MinMaxScaler()
    data_x_scaled = scaler_x.fit_transform(x)
    return data_x_scaled, scaler_x

def preprocess_data_y(y):
    scaler_y = MinMaxScaler()
    data_y_scaled = scaler_y.fit_transform(y)
    return data_y_scaled, scaler_y

def split_data(data_x_scaled, data_y_scaled, train_size):
    X_train, X_test, y_train, y_test = train_test_split(data_x_scaled, data_y_scaled, train_size=train_size, shuffle=False)
    return X_train, X_test, y_train, y_test

def future_predict(model, data, scaler_x, scaler_y):
    last_data = data[['Close']].tail(50)
    # current_features = data[['Close']].tail(100)
    future_prediction = []
    current_features = scaler_y.transform(last_data)

    for i in range(20):
        next_day_pred = model.predict(current_features[-50].reshape(1, -1))
        future_prediction.append(next_day_pred)
        next_day_features = np.roll(current_features, -1)
        next_day_features[-1] = np.append(next_day_features[-2, 1:], next_day_pred.reshape(1,-50))
        current_features = next_day_features

    future_prediction = scaler_y.inverse_transform(np.array(future_prediction).reshape(-1, 1)).flatten()

    # Prepare data for dynamic prediction
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.date
    last_date = data['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 21)]
    

    return future_prediction, future_dates

def train_and_predict(data, split_ratio):
    x, y = create_data(data)
    data_x_scaled, scaler_x = preprocess_data_x(x)
    data_y_scaled, scaler_y = preprocess_data_y(y)
    X_train, X_test, y_train, y_test = split_data(data_x_scaled, data_y_scaled, train_size=split_ratio)

    model = SVR(kernel='rbf', C=100, gamma=0.1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    future_prediction, future_dates = future_predict(model, data, scaler_x, scaler_y)

    # Evaluate the Model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    target_range = np.max(y_test) - np.min(y_test)
    accuracy = (1 - (rmse / target_range)) * 100


    return predictions, y_test, accuracy, rmse, future_prediction, future_dates