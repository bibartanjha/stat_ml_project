# stat_ml_project

## Updates

### Apr 3 from Bib
Feature extraction is done. I've extracted all the following features from the audio files

- Chroma: A spectrogram showing intensity of frequencies over time.
- Tonnetz: Also known as tonal centroids, captures musical harmony information. 
- Mel-Frequency Cepstrum Coefficients (MFCC): Commonly used feature in speech  recognition and music information retrieval.
- Spectral Centroid: Describes the ratio between magnitudes of peaks and valleys within  sub-bands of a frequency spectrum.
- Spectral Bandwidth: Bandwidth is the measurement of the difference in wavelength at (usually) half the intensity.
- Spectral Rolloff: Measure of where a specified percentage of total spectral energy lies.
- Root Mean Square Energy (RMSE): As the name indicates, square root of the mean of squares of signal values over a window.

This can be seen in the folder "feature_extraction". All my code to extract the features is in data_collection_updated.ipynb

I ran the code, and it created 3 datasets
- dataOrig.csv : This is the original dataset
- dataForHarmonicAudio.csv : This is the original dataset plus some pre-processing. I used a liberosa function to extract only the harmonies from each of the audio files, so it removes any percussion and other noise
- dataForMFCCDeltas.csv : This is another dataset, where we're using the deltas of the mfcc feature as the inputs.

We can use any of the 3 datasets above in our future code for our ML models. 

I've also written some basic python code in modeltraining/modeltraining.py. Ideally, our ML model training code will go in this file.


### April 6th from Laurenz
Initial Model structure is there.

- ModelEvaluator class that takes a list of models to evaluate and data.
- Each model must be a BaseModel implementation. 
- BaseModel take a model that needs to implement the fit() and predict() methods.

See modeltraining.py for an example how to use it. 