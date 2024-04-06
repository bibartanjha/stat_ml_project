import pandas as pd
from models import (
    ModelEvaluator,
    LinearRegressionModel, LogisticRegressionModel, RandomForestClassifierModel, SVCModel,
    LassoModel, RidgeModel, ElasticNetModel, ElasticNetCVModel, SVRModel, LinearSVCModel,
    DecisionTreeClassifierModel, ExtraTreeClassifierModel, KNeighborsClassifierModel,
    GradientBoostingClassifierModel, KMeansModel, GaussianMixtureModel,
    # These don't work at the moment:
    # DenseModel, RNNModel, LSTMModel, GRUModel
)


if __name__ == '__main__':
    # We can use any of the 3 data sets above to feed into our models
    data = pd.read_csv(
        "../feature_extraction/dataOrig.csv",
        usecols=lambda column: column != 'filename'
    )

    X = data.iloc[:, :-1]  # the last column is the label
    y = data.iloc[:, -1]  # the label

    me = ModelEvaluator(models=[
        LinearRegressionModel, LogisticRegressionModel, RandomForestClassifierModel, SVCModel,
        LassoModel, RidgeModel, ElasticNetModel, ElasticNetCVModel, SVRModel, LinearSVCModel,
        DecisionTreeClassifierModel, ExtraTreeClassifierModel, KNeighborsClassifierModel,
        GradientBoostingClassifierModel, KMeansModel, GaussianMixtureModel,
        # DenseModel, RNNModel, LSTMModel, GRUModel
    ], X=X, y=y, test_size=0.3, random_state=50)
    me.evaluate_models()
    me.plot_results(accuracy=True, mse=False, training_time=False, prediction_time=False)

    data = pd.read_csv(
        "../feature_extraction/dataForHarmonicAudio.csv",
        usecols=lambda column: column != 'filename'
    )

    X = data.iloc[:, :-1]  # the last column is the label
    y = data.iloc[:, -1]  # the label
    me = ModelEvaluator(models=[
        LinearRegressionModel, LogisticRegressionModel, RandomForestClassifierModel, SVCModel,
        LassoModel, RidgeModel, ElasticNetModel, ElasticNetCVModel, SVRModel, LinearSVCModel,
        DecisionTreeClassifierModel, ExtraTreeClassifierModel, KNeighborsClassifierModel,
        GradientBoostingClassifierModel, KMeansModel, GaussianMixtureModel,
        # DenseModel, RNNModel, LSTMModel, GRUModel
    ], X=X, y=y, test_size=0.3, random_state=50)
    me.evaluate_models()
    me.plot_results(accuracy=True, mse=False, training_time=False, prediction_time=False)

    data = pd.read_csv(
        "../feature_extraction/dataForMFCCDeltas.csv",
        usecols=lambda column: column != 'filename'
    )
    X = data.iloc[:, :-1]  # the last column is the label
    y = data.iloc[:, -1]  # the label
    me = ModelEvaluator(models=[
        LinearRegressionModel, LogisticRegressionModel, RandomForestClassifierModel, SVCModel,
        LassoModel, RidgeModel, ElasticNetModel, ElasticNetCVModel, SVRModel, LinearSVCModel,
        DecisionTreeClassifierModel, ExtraTreeClassifierModel, KNeighborsClassifierModel,
        GradientBoostingClassifierModel, KMeansModel, GaussianMixtureModel,
        # DenseModel, RNNModel, LSTMModel, GRUModel
    ], X=X, y=y, test_size=0.3, random_state=50)
    me.evaluate_models()
    me.plot_results(accuracy=True, mse=False, training_time=False, prediction_time=False)

    print('done')


