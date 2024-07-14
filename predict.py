import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split

def variable_choose(data):
    x = data[['Open','High','Low','Close']]
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

class SVR_Model:
    def __init__(self, gamma, C, epsilon = 0.01, max_iter=1000):
        self.gamma = gamma  # Parameter untuk kernel RBF
        self.C = C          # Parameter regularisasi
        self.epsilon = epsilon
        self.alpha = None   # Koefisien alpha akan disimpan setelah pelatihan
        self.alpha_star = None
        self.X_train = None # Menyimpan data pelatihan untuk digunakan dalam prediksi
        self.max_iter = max_iter

    def rbf_kernel(self, X, X2):
        """ Calculate the RBF kernel between two datasets. """
        X = np.asarray(X)
        X2 = np.asarray(X2)
        sq_dists = np.sum((X[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
        return np.exp(-self.gamma * sq_dists)

    def train(self, X_train, y_train):
            self.X_train = X_train
            n_samples = X_train.shape[0]
            y_train = np.asarray(y_train).flatten()
            self.alpha = np.zeros(n_samples)
            self.alpha_star = np.zeros(n_samples)
            K_train = self.rbf_kernel(X_train, X_train)

            for iteration in range(self.max_iter):
                alpha_prev = np.copy(self.alpha)
                alpha_star_prev = np.copy(self.alpha_star)
                
                for i in range(n_samples):
                    Ei = y_train[i] - np.dot((self.alpha - self.alpha_star), K_train[i])
                    delta_alpha = min(max(self.gamma * (Ei - self.epsilon), -self.alpha[i]), self.C - self.alpha[i])
                    delta_alpha_star = min(max(self.gamma * (-Ei - self.epsilon), -self.alpha_star[i]), self.C - self.alpha_star[i])
    
                    self.alpha[i] += delta_alpha
                    self.alpha_star[i] += delta_alpha_star

                # Check convergence
                alpha_diff = np.linalg.norm(self.alpha - alpha_prev)
                alpha_star_diff = np.linalg.norm(self.alpha_star - alpha_star_prev)
                print(f"Iteration {iteration}, norm_alpha_diff: {alpha_diff}, norm_alpha_star_diff: {alpha_star_diff}")
                if alpha_diff < self.epsilon and alpha_star_diff < self.epsilon: 
                    print(f"Converged after {iteration} iterations.")
                    break
                

    def predict(self, X_test):
        """ Predict using the trained SVR model. """
        if self.alpha is None or self.alpha_star is None:
            raise Exception("Model has not been trained yet.")
        K_test = self.rbf_kernel(X_test, self.X_train)
        return K_test.dot(self.alpha - self.alpha_star)
    
    def predict_future(self, current_features):
        """Predict using the trained SVR model."""
        if self.alpha is None or self.alpha_star is None:
            raise Exception("Model has not been trained yet.")
        K_future = self.rbf_kernel(current_features, self.X_train)
        alpha_coef = self.alpha - self.alpha_star
        return np.dot(K_future, alpha_coef)

def future_predict(model, data, y_scaled):
    current_features = y_scaled[-4:].values.reshape(1,-1)
    future_predictions = []

    for i in range(7):
        next_day_pred = model.predict_future(current_features)
        future_predictions.append(next_day_pred)
        next_day_features = np.roll(current_features, -1, axis=0)
        next_day_features[-1] = np.append(next_day_features[-1, 1:], next_day_pred)
        current_features = next_day_features

    # Prepare data for dynamic prediction
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.date
    last_date = data['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

    if future_predictions[-1] > future_predictions[0]:
        price_movement = "Rise :arrow_upper_right:"
    elif future_predictions[-1] < future_predictions[0]:
        price_movement = "Drop :arrow_lower_right:"
    

    return future_predictions, future_dates, price_movement


def train_and_predict(data, split_ratio, gamma_value, c_value):
    x, y = variable_choose(data)
    x_scaled, min_x, max_x = preprocess_data_x(x)
    y_scaled, min_y, max_y = preprocess_data_y(y)
    X_train, X_test, y_train, y_test = split_data(x_scaled, y_scaled, train_size=split_ratio)

    model = SVR_Model (gamma=gamma_value, C=c_value)
    model.train(X_train, y_train)
    predictions = model.predict(X_test)

    future_predictions, future_dates, price_movement = future_predict(model, data, y_scaled)

    predictions = np.array(predictions).flatten()
    y_test = np.array(y_test).flatten()
    future_predictions = np.array(future_predictions).flatten()

    min_y = float(min_y.iloc[0])
    max_y = float(max_y.iloc[0])

    # Denormalisasi data untuk mengembalikan ke nilai asli
    predictions = predictions * (max_y - min_y) + min_y
    y_test = y_test * (max_y - min_y) + min_y
    future_predictions = future_predictions * (max_y - min_y) + min_y

    #Evaluate the Model
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    accuracy = (100 - mape)

 

   

    return predictions, y_test, accuracy, mape, future_predictions, future_dates, price_movement