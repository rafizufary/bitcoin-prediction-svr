import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import timedelta

def preprocess_data_x(data):
    data_x = data[['Close']]
    min_x = data_x.min()
    max_x = data_x.max()
    data_x_scaled = (data_x - min_x) / (max_x - min_x)
    return data_x_scaled.values, min_x, max_x  # Ubah ke array NumPy

def preprocess_data_y(data):
    data_y = data[['Close']]
    min_y = data_y.min()
    max_y = data_y.max()
    data_y_scaled = (data_y - min_y) / (max_y - min_y)
    return data_y_scaled.values, min_y, max_y,  # Ubah ke array NumPy

def split_data(data_x_scaled, data_y_scaled):
    X = data_x_scaled
    y = data_y_scaled
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def compute_kernel_matrix(X, X2, gamma):
    n_samples, n_samples2 = X.shape[0], X2.shape[0]
    K = np.zeros((n_samples, n_samples2))
    for i in range(n_samples):
        for j in range(n_samples2):
            K[i, j] = rbf_kernel(X[i], X2[j], gamma)  # Perbaikan pengindeksan
    return K

def rbf_regression(X_train, y_train, X_test, gamma, C):
    lambda_reg = 1 / C
    K_train = compute_kernel_matrix(X_train, X_train, gamma)
    K_train += lambda_reg * np.eye(K_train.shape[0])
    alpha = np.linalg.solve(K_train, y_train)
    K_test = compute_kernel_matrix(X_test, X_train, gamma)
    y_pred = K_test.dot(alpha)
    return y_pred

def train_model(X_train, y_train, gamma=0.1, C=10):
    return rbf_regression(X_train, y_train, X_train, gamma, C)

def evaluate_model(X_train, y_train, X_test, y_test, gamma=0.1, C=100):
    predictions = rbf_regression(X_train, y_train, X_test, gamma, C)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    target_range = np.max(y_test) - np.min(y_test)
    accuracy = (1 - (rmse / target_range)) * 100
    return predictions, y_test, accuracy, rmse

def train_and_predict(data):
    data_x_scaled, min_x, max_x = preprocess_data_x(data)
    data_y_scaled, min_y, max_y = preprocess_data_y(data)
    X_train, X_test, y_train, y_test = split_data(data_x_scaled, data_y_scaled)
    model = train_model(X_train, y_train)
    predictions, y_test, accuracy, rmse = evaluate_model(X_train, y_train, X_test, y_test)
    
    # Konversi ke NumPy array jika belum
    predictions = np.array(predictions).flatten()  # Mengubah ke array dan memastikan format flat
    y_test = np.array(y_test).flatten()

    # Pastikan min_y dan max_y adalah skalar, tidak pd.Series atau pd.DataFrame
    min_y = float(min_y.iloc[0])  # Konversi ke float jika belum
    max_y = float(max_y.iloc[0])

    # Skala ulang ke rentang data asli
    predictions = predictions * (max_y - min_y) + min_y
    y_test = y_test * (max_y - min_y) + min_y

    return predictions, y_test, accuracy, rmse