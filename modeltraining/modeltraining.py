import pandas as pd
from sklearn.model_selection import train_test_split
from models import (
    LinearRegressionModel, LogisticRegressionModel, SVCModel,
    RandomForestClassifierModel, ModelEvaluator
)


if __name__ == '__main__':
    # We can use any of the 3 data sets above to feed into our models
    data = pd.read_csv(
        "../feature_extraction/dataOrig.csv",
        usecols=lambda column: column != 'filename'
    )
    # dataWithOnlyHarmonicsOfAudioFiles = pd.read_csv(
    #     "../feature_extraction/dataForHarmonicAudio.csv",
    #     usecols=lambda column: column != 'filename'
    # )
    # dataWithDeltasOfMFCCs = pd.read_csv(
    #     "../feature_extraction/dataForMFCCDeltas.csv",
    #     usecols=lambda column: column != 'filename'
    # )

    X = data.iloc[:, :-1]  # the last column is the label
    y = data.iloc[:, -1]  # the label

    me = ModelEvaluator(models=[
        LinearRegressionModel, LogisticRegressionModel,
        SVCModel, RandomForestClassifierModel
    ], X=X, y=y)
    me.evaluate_models()
    me.plot_results()

    print('done')


