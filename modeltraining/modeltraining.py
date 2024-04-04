import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../feature_extraction/dataOrig.csv", usecols=lambda column: column != 'filename')
dataWithOnlyHarmonicsOfAudioFiles = pd.read_csv("../feature_extraction/dataForHarmonicAudio.csv", usecols=lambda column: column != 'filename')
dataWithDeltasOfMFCCs = pd.read_csv("../feature_extraction/dataForMFCCDeltas.csv", usecols = lambda column: column != 'filename')

# We can use any of the 3 data sets above to feed into our models


X = data.iloc[:, :-1] # the last column is the label
y = data.iloc[:, -1] # the label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


