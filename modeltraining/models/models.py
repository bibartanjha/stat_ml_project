import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC


class BaseModel:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.le = LabelEncoder()
        self.y_train = self.le.fit_transform(y_train)
        self.y_test = self.le.fit_transform(y_test)
        self.prediction_time = 0
        self.predictions = 0
        self.training_time = 0
        self.mse_values = []
        self.correct_predictions = 0
        self.incorrect_predictions = 0
        self.fit()

    def prediction_performance(self):
        return self.prediction_time / self.predictions

    def training_performance(self):
        return self.training_time / self.X_train.shape[0]

    def mean_squared_error(self):
        return np.mean(self.mse_values)

    def accuracy(self):
        return self.correct_predictions / (self.correct_predictions + self.incorrect_predictions)

    def fit(self):
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.training_time += time.time() - start_time

    def predict(self, X, y=None, return_code=False):
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time += time.time() - start_time
        self.predictions += X.shape[0]

        if y is not None:
            self.mse_values.append(mean_squared_error(y, predictions))

        rounded_predictions = np.round(predictions).astype(int)
        clipped_predictions = np.clip(rounded_predictions, 0, len(self.le.classes_) - 1)
        return clipped_predictions if return_code else self.le.inverse_transform(clipped_predictions)

    def evaluate(self, X_test=None, y_test=None):
        X_test = X_test if X_test is not None else self.X_test
        y_test = y_test if y_test is not None else self.y_test
        predictions = self.predict(X_test, y_test, return_code=True)
        correct_predictions = np.equal(predictions, y_test)
        self.correct_predictions += np.sum(correct_predictions)
        self.incorrect_predictions += len(correct_predictions) - np.sum(correct_predictions)
        accuracy = np.sum(correct_predictions) / len(correct_predictions)
        return accuracy, self.mse_values[-1]

    def plot_performance(self):
        predictions = self.predict(self.X_test)
        plt.scatter(self.y_test, predictions)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Model Performance')
        plt.grid(True)
        plt.show()


class LinearRegressionModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(LinearRegression(), X_train, X_test, y_train, y_test)


class LogisticRegressionModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(LogisticRegression(), X_train, X_test, y_train, y_test)


class RandomForestClassifierModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(RandomForestClassifier(), X_train, X_test, y_train, y_test)


class SVCModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(SVC(), X_train, X_test, y_train, y_test)