import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet, ElasticNetCV
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers import Dense, SimpleRNN, LSTM, GRU


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
        super().__init__(LogisticRegression(max_iter=1500), X_train, X_test, y_train, y_test)


class RandomForestClassifierModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(
            RandomForestClassifier(**{
                'max_depth': 20, 'max_features': 4, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500
            }), X_train, X_test, y_train, y_test
        )


class SVCModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(SVC(kernel='linear', degree=3, C=16, random_state=42), X_train, X_test, y_train, y_test)


class LassoModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(Lasso(), X_train, X_test, y_train, y_test)


class RidgeModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(Ridge(), X_train, X_test, y_train, y_test)


class ElasticNetModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(ElasticNet(), X_train, X_test, y_train, y_test)


class ElasticNetCVModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(ElasticNetCV(), X_train, X_test, y_train, y_test)


class SVRModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(SVR(), X_train, X_test, y_train, y_test)


class LinearSVCModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(LinearSVC(), X_train, X_test, y_train, y_test)


class DecisionTreeClassifierModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(DecisionTreeClassifier(
            **{'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 10}
        ), X_train, X_test, y_train, y_test)


class ExtraTreeClassifierModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(ExtraTreeClassifier(), X_train, X_test, y_train, y_test)


class KNeighborsClassifierModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(KNeighborsClassifier(), X_train, X_test, y_train, y_test)


class GradientBoostingClassifierModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(GradientBoostingClassifier(), X_train, X_test, y_train, y_test)


class KMeansModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(KMeans(), X_train, X_test, y_train, y_test)


class GaussianMixtureModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(GaussianMixture(), X_train, X_test, y_train, y_test)

#
# class DenseModel(BaseModel):
#     def __init__(self, X_train, X_test, y_train, y_test):
#         model = Sequential()
#         model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
#         model.add(Dense(64, activation='relu'))
#         model.add(Dense(1, activation='sigmoid'))
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#         super().__init__(model, X_train, X_test, y_train, y_test)
#
#
# class RNNModel(BaseModel):
#     def __init__(self, X_train, X_test, y_train, y_test):
#         model = Sequential()
#         model.add(SimpleRNN(128, activation='relu', input_shape=(X_train.shape[1], 1)))
#         model.add(Dense(64, activation='relu'))
#         model.add(Dense(1, activation='sigmoid'))
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#         super().__init__(model, X_train, X_test, y_train, y_test)
#
#
# class LSTMModel(BaseModel):
#     def __init__(self, X_train, X_test, y_train, y_test):
#         model = Sequential()
#         model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], 1)))
#         model.add(Dense(64, activation='relu'))
#         model.add(Dense(1, activation='sigmoid'))
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#         super().__init__(model, X_train, X_test, y_train, y_test)
#
#
# class GRUModel(BaseModel):
#     def __init__(self, X_train, X_test, y_train, y_test):
#         model = Sequential()
#         model.add(GRU(128, activation='relu', input_shape=(X_train.shape[1], 1)))
#         model.add(Dense(64, activation='relu'))
#         model.add(Dense(1, activation='sigmoid'))
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#         super().__init__(model, X_train, X_test, y_train, y_test)
