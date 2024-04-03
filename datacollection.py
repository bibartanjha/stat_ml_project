'''
To run the code in this file, make sure to run these first:
pip install librosa
pip install librosa soundfile
pip install setuptools
pip install pandas
'''


import os
import librosa
import numpy as np
import pandas as pd

pathToAllFiles = 'genres'
allGenres = os.listdir(pathToAllFiles)

allRows = []
for genre in allGenres:
    print(genre)
    pathToGenreFiles = os.path.join(pathToAllFiles, genre)
    for file in os.listdir(pathToGenreFiles):
        pathToFile = os.path.join(pathToGenreFiles, file)

        try:
            audio, sr = librosa.load(pathToFile, res_type='kaiser_fast')
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)

            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)

            row = [file] + chroma_mean.tolist() + tonnetz_mean.tolist() + [genre]

            allRows.append(row)

        except Exception as e:
            print("Error processing file:", pathToFile)
            continue

columns = ['filename'] + [f'chroma_{i}' for i in range(len(chroma_mean))] + [f'tonnetz_{i}' for i in range(len(tonnetz_mean))] + ['genre']
df = pd.DataFrame(allRows, columns=columns)
csv_file = 'data.csv'
df.to_csv(csv_file, index=False)
print("Done")
