import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def prepare_data(data):
    x = data[['Open','High','Low','Volume']]
    y = data[['Close']]

    return x, y

def preprocess_data_x(x):
    min_x = x.min()
    max_x = x.max()
    x_scaled = (x - min_x) / (max_x - min_x)

    return x_scaled, min_x, max_x

def preprocess_data_y(y):
    min_y = y.min()
    max_y = y.max()
    y_scaled = (y - min_y) / (max_y - min_y)

    return y_scaled, min_y, max_y

def split_data(x_scaled, y_scaled, train_size):
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, train_size=train_size, shuffle=False)
    return X_train, X_test, y_train, y_test

# def rbf_kernel(X, X2, gamma):
#     X = np.asarray(X)
#     X2 = np.asarray(X2)
#     sq_dists = np.sum((X[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
#     return np.exp(-gamma * sq_dists)

# def svr_train(X_train, y_train, gamma, C):
#     """ Train an SVR model. """
#     lambda_reg = 1 / C
#     K_train = rbf_kernel(X_train, X_train, gamma)
#     K_train += lambda_reg * np.eye(K_train.shape[0])
#     alpha = np.linalg.solve(K_train, y_train)
#     return alpha
# def svr_predict(X_train, X_test, alpha, gamma):
    """ Predict using the trained SVR model. """
    K_test = rbf_kernel(X_test, X_train, gamma)
    y_pred = K_test.dot(alpha)
    return y_pred

class SVR_Model:
    def __init__(self, gamma, C):
        self.gamma = gamma  # Parameter untuk kernel RBF
        self.C = C          # Parameter regularisasi
        self.alpha = None   # Koefisien alpha akan disimpan setelah pelatihan
        self.X_train = None # Menyimpan data pelatihan untuk digunakan dalam prediksi

    def rbf_kernel(self, X, X2):
        """ Calculate the RBF kernel between two datasets. """
        X = np.asarray(X)
        X2 = np.asarray(X2)
        sq_dists = np.sum((X[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
        return np.exp(-self.gamma * sq_dists)

    def train(self, X_train, y_train):
        """ Train the SVR model using the provided training dataset. """
        self.X_train = X_train  # Menyimpan X_train untuk digunakan di predict
        lambda_reg = 1 / self.C
        K_train = self.rbf_kernel(X_train, X_train)
        K_train += lambda_reg * np.eye(K_train.shape[0])
        self.alpha = np.linalg.solve(K_train, y_train)

    def predict(self, X_test):
        """ Predict using the trained SVR model. """
        if self.alpha is None:
            raise Exception("Model has not been trained yet.")
        K_test = self.rbf_kernel(X_test, self.X_train)
        return K_test.dot(self.alpha)

def future_predict(model, data, x_scaled):
    current_features = x_scaled[-20:]
    future_predictions = []
    

    for i in range(7):
        next_day_pred = model.predict(current_features)[0, 0]
        future_predictions.append(next_day_pred)
        next_day_features = np.roll(current_features, -1, axis=0)
        next_day_features[-1] = np.append(next_day_features[-2, 1:], next_day_pred)
        current_features = next_day_features

    # Prepare data for dynamic prediction
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.date
    last_date = data['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    

    return future_predictions, future_dates


def train_and_predict(data, split_ratio, gamma_value, c_value):
    x, y = prepare_data(data)
    x_scaled, min_x, max_x = preprocess_data_x(x)
    y_scaled, min_y, max_y = preprocess_data_y(y)
    X_train, X_test, y_train, y_test = split_data(x_scaled, y_scaled, train_size=split_ratio)

    model = SVR_Model(gamma=gamma_value, C=c_value)
    model.train(X_train, y_train)
    predictions = model.predict(X_test)

    future_predictions, future_dates = future_predict(model, data, x_scaled)


    #Evaluate the Model

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions) * 100
    target_range = np.max(y_test) - np.min(y_test)
    # accuracy = (1 - (rmse / target_range)) * 100
    accuracy = (100 - mape)

 

    predictions = np.array(predictions).flatten()  # Mengubah ke array dan memastikan format flat
    y_test = np.array(y_test).flatten()
    future_predictions = np.array(future_predictions).flatten()

    min_y = float(min_y.iloc[0])
    max_y = float(max_y.iloc[0])

    # Denormalisasi data untuk mengembalikan ke nilai asli
    predictions = predictions * (max_y - min_y) + min_y
    y_test = y_test * (max_y - min_y) + min_y
    future_predictions = future_predictions * (max_y - min_y) + min_y

    return predictions, y_test, accuracy, rmse, future_predictions, future_dates