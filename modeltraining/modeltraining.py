import os

import pandas as pd
from sklearn import preprocessing

from models import (
    ModelEvaluator,
    LinearRegressionModel, LogisticRegressionModel, RandomForestClassifierModel, SVCModel,
    LassoModel, RidgeModel, ElasticNetModel, ElasticNetCVModel, SVRModel, LinearSVCModel,
    DecisionTreeClassifierModel, ExtraTreeClassifierModel, KNeighborsClassifierModel,
    GradientBoostingClassifierModel, KMeansModel, GaussianMixtureModel,
    # These don't work at the moment:
    # DenseModel, RNNModel, LSTMModel, GRUModel
)


def eval_models(directory, scalers):
    filenames = []
    for filename in os.listdir(directory):
        filename = os.path.join(directory, filename)
        if os.path.isfile(filename):
            filenames.append(filename)

    me = ModelEvaluator(models=[
        LogisticRegressionModel, RandomForestClassifierModel, SVCModel, LinearSVCModel,
        DecisionTreeClassifierModel, ExtraTreeClassifierModel, KNeighborsClassifierModel,
        GradientBoostingClassifierModel,
        # DenseModel, RNNModel, LSTMModel, GRUModel
    ], filenames=filenames, scalers=scalers, test_size=0.3, random_state=42)
    return me.plot_results(accuracy=True, mse=False, training_time=False, prediction_time=False), me.results


if __name__ == '__main__':
    SCALERS = [
        preprocessing.StandardScaler, preprocessing.MinMaxScaler,
        preprocessing.MaxAbsScaler, preprocessing.RobustScaler
    ]
    best_results, results = eval_models('../allDataSets/dataOrigFromLibrosa/preprocessingOutputCSVs/AllGenres', SCALERS)
    for res in results:
        print(res)
    print('done')


        # # Check if the current item is a file
        # filename = os.path.join(directory, filename)
        # if os.path.isfile(filename):
        #     # data = pd.read_csv(
        #     #     filename, usecols=lambda column: column != 'filename'
        #     # )
        #     #
        #     # X = data.iloc[:, :-1]  # the last column is the label
        #     # y = data.iloc[:, -1]  # the label
        #
        #     me = ModelEvaluator(models=[
        #         LogisticRegressionModel, RandomForestClassifierModel, SVCModel, LinearSVCModel,
        #         DecisionTreeClassifierModel, ExtraTreeClassifierModel, KNeighborsClassifierModel,
        #         GradientBoostingClassifierModel,
        #         # DenseModel, RNNModel, LSTMModel, GRUModel
        #     ], X=X, y=y, test_size=0.3, random_state=50)
        #     me.evaluate_models()
        #     me.plot_results(accuracy=True, mse=False, training_time=False, prediction_time=False, name='Original')

    # data = pd.read_csv(
    #     "../feature_extraction/dataOrig.csv",
    #     usecols=lambda column: column != 'filename'
    # )
    #
    # X = data.iloc[:, :-1]  # the last column is the label
    # y = data.iloc[:, -1]  # the label
    #
    # me = ModelEvaluator(models=[
    #     LogisticRegressionModel, RandomForestClassifierModel, SVCModel, LinearSVCModel,
    #     DecisionTreeClassifierModel, ExtraTreeClassifierModel, KNeighborsClassifierModel,
    #     GradientBoostingClassifierModel,
    #     # DenseModel, RNNModel, LSTMModel, GRUModel
    # ], X=X, y=y, test_size=0.3, random_state=50)
    # me.evaluate_models()
    # me.plot_results(accuracy=True, mse=False, training_time=False, prediction_time=False, name='Original')
    #
    # data = pd.read_csv(
    #     "../feature_extraction/dataForHarmonicAudio.csv",
    #     usecols=lambda column: column != 'filename'
    # )
    #
    # X = data.iloc[:, :-1]  # the last column is the label
    # y = data.iloc[:, -1]  # the label
    # me = ModelEvaluator(models=[
    #     LinearRegressionModel, LogisticRegressionModel, RandomForestClassifierModel, SVCModel,
    #     LassoModel, RidgeModel, ElasticNetModel, ElasticNetCVModel, SVRModel, LinearSVCModel,
    #     DecisionTreeClassifierModel, ExtraTreeClassifierModel, KNeighborsClassifierModel,
    #     GradientBoostingClassifierModel, KMeansModel, GaussianMixtureModel,
    #     # DenseModel, RNNModel, LSTMModel, GRUModel
    # ], X=X, y=y, test_size=0.3, random_state=50)
    # me.evaluate_models()
    # me.plot_results(accuracy=True, mse=False, training_time=False, prediction_time=False, name='HarmonicAudio')
    #
    # data = pd.read_csv(
    #     "../feature_extraction/dataForMFCCDeltas.csv",
    #     usecols=lambda column: column != 'filename'
    # )
    # X = data.iloc[:, :-1]  # the last column is the label
    # y = data.iloc[:, -1]  # the label
    # me = ModelEvaluator(models=[
    #     LinearRegressionModel, LogisticRegressionModel, RandomForestClassifierModel, SVCModel,
    #     LassoModel, RidgeModel, ElasticNetModel, ElasticNetCVModel, SVRModel, LinearSVCModel,
    #     DecisionTreeClassifierModel, ExtraTreeClassifierModel, KNeighborsClassifierModel,
    #     GradientBoostingClassifierModel, KMeansModel, GaussianMixtureModel,
    #     # DenseModel, RNNModel, LSTMModel, GRUModel
    # ], X=X, y=y, test_size=0.3, random_state=50)
    # me.evaluate_models()
    # me.plot_results(accuracy=True, mse=False, training_time=False, prediction_time=False, name='MFCCDeltas')
    #
    # print('done')


