import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_data(data):
    data_1 = data[['Close']]
    data_2 = data[['Close']]
    return data_1, data_2

def preprocess_data_x(data_1):
    data_x = data_1
    min_x = data_x.min()
    max_x = data_x.max()
    data_x_scaled = (data_x - min_x) / (max_x - min_x)
    return data_x_scaled

def preprocess_data_y(data_2):
    data_y = data_2
    min_y = data_y.min()
    max_y = data_y.max()
    data_y_scaled = (data_y - min_y) / (max_y - min_y)
    return data_y_scaled, min_y, max_y

def split_data(data_x_scaled, data_y_scaled, train_size):
    x = data_x_scaled
    y = data_y_scaled
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=train_size, shuffle=False)
    return X_train, X_test, y_train, y_test

def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def compute_kernel_matrix(X, X2, gamma):
    X = np.asarray(X)
    X2 = np.asarray(X2)
    sq_dists = np.sum((X[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
    return np.exp(-gamma * sq_dists)

def regression(X_train, y_train, X_test, gamma, C):
    lambda_reg = 1 / C
    K_train = compute_kernel_matrix(X_train, X_train, gamma)
    K_train += lambda_reg * np.eye(K_train.shape[0])
    alpha = np.linalg.solve(K_train, y_train)
    K_test = compute_kernel_matrix(X_test, X_train, gamma)
    y_pred = K_test.dot(alpha)
    return y_pred, alpha


def train_and_predict(data, split_ratio, gamma=0.1, C=100):
    x, y = create_data(data)
    data_x_scaled = preprocess_data_x(x)
    data_y_scaled, min_y, max_y = preprocess_data_y(y)
    X_train, X_test, y_train, y_test = split_data(data_x_scaled, data_y_scaled, train_size=split_ratio)
    predictions = regression(X_train, y_train, y_test, gamma, C)[0]
    # svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
    # svr_model.fit(X_train, y_train)
    # predictions = svr_model.predict(X_test)
    
    future_days = 7
    future_x = np.array([len(y_test) + i for i in range(future_days)]).reshape(-1, 1)
    future_x_scaled = preprocess_data_x(future_x)
    future_prediction, _ = regression(X_train, y_train, future_x_scaled, gamma, C)
    # future_pred = svr_model.predict(future_x_scaled)
    
    
    #Evaluate the Model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    target_range = np.max(y_test) - np.min(y_test)
    accuracy = (1 - (rmse / target_range)) * 100
    
    # Konversi ke NumPy array jika belum
    predictions = np.array(predictions).flatten()  # Mengubah ke array dan memastikan format flat
    y_test = np.array(y_test).flatten()

    # Pastikan min_y dan max_y adalah skalar, tidak pd.Series atau pd.DataFrame
    min_y = float(min_y.iloc[0])  # Konversi ke float jika belum
    max_y = float(max_y.iloc[0])

    # Skala ulang ke rentang data asli
    predictions = predictions * (max_y - min_y) + min_y
    y_test = y_test * (max_y - min_y) + min_y
    future_prediction = (np.array(future_prediction).flatten() * (max_y - min_y)) + min_y

    
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.date
    last_date = data['Date'].iloc[0]
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]


    return predictions, y_test, accuracy, rmse, future_prediction, future_dates