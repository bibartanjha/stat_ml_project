import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from .models import (
    LinearRegressionModel, LogisticRegressionModel, RandomForestClassifierModel, SVCModel,
    LassoModel, RidgeModel, ElasticNetModel, ElasticNetCVModel, SVRModel, LinearSVCModel,
    DecisionTreeClassifierModel, ExtraTreeClassifierModel, KNeighborsClassifierModel,
    GradientBoostingClassifierModel, KMeansModel, GaussianMixtureModel, DenseModel, RNNModel,
    LSTMModel, GRUModel
)


class ModelEvaluator:
    def __init__(self, models, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.models = [model(X_train, X_test, y_train, y_test) for model in models]  # Initialize models
        self.results = []

    def evaluate_models(self):
        for model in self.models:
            test_accuracy, test_mse = model.evaluate()
            train_accuracy, train_mse = model.evaluate(model.X_train, model.y_train)
            self.results.append({
                'model': model.__class__.__name__,
                'accuracy': model.accuracy(),
                'test_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'mse': model.mean_squared_error(),
                'test_mse': test_mse,
                'train_mse': train_mse,
                'training_time': model.training_performance(),
                'prediction_time': model.prediction_performance()
            })

    def plot_results(self, accuracy=True, mse=True, training_time=True, prediction_time=True):
        self.results.sort(key=lambda x: x['test_accuracy'])
        model_names = [result['model'] for result in self.results]
        accuracies = [result['accuracy'] for result in self.results]
        test_accuracies = [result['test_accuracy'] for result in self.results]
        train_accuracies = [result['train_accuracy'] for result in self.results]
        mses = [result['mse'] for result in self.results]
        test_mses = [result['test_mse'] for result in self.results]
        train_mses = [result['train_mse'] for result in self.results]
        prediction_times = [result['prediction_time'] for result in self.results]
        training_times = [result['training_time'] for result in self.results]

        if accuracy:
            # Plot accuracies
            plt.figure(figsize=(12, 6))
            bar_width = 0.25
            index = np.arange(len(model_names))
            # bar1 = plt.bar(index, accuracies, bar_width, label='Accuracy', color='blue')
            bar2 = plt.bar(index + bar_width, test_accuracies, bar_width, label='Test Accuracy', color='green')
            bar3 = plt.bar(index + 2 * bar_width, train_accuracies, bar_width, label='Train Accuracy', color='orange')

            # Add the data value on top of each bar
            for bars in [bar2]:
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='bottom')

            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Metrics for Different Models')
            plt.xticks(index + bar_width, model_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        if mse:
            # Plot mses
            plt.figure(figsize=(12, 6))
            plt.bar(index, mses, bar_width, label='MSE', color='blue')
            plt.bar(index + bar_width, test_mses, bar_width, label='Test MSE', color='green')
            plt.bar(index + 2 * bar_width, train_mses, bar_width, label='Train MSE', color='orange')
            plt.xlabel('Models')
            plt.ylabel('MSE')
            plt.title('Mean Squared Error Metrics for Different Models')
            plt.xticks(index + bar_width, model_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        if training_time:
            # Plot training time
            plt.figure(figsize=(12, 6))
            plt.bar(model_names, training_times, color='blue')
            plt.xlabel('Models')
            plt.ylabel('Training Time (seconds)')
            plt.title('Training Time for Different Models')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        if prediction_time:
            # Plot prediction time
            plt.figure(figsize=(12, 6))
            plt.bar(model_names, prediction_times, color='green')
            plt.xlabel('Models')
            plt.ylabel('Prediction Time (seconds)')
            plt.title('Prediction Time for Different Models')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()


