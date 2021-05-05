import pandas as pd
from pandas import Timestamp
from datetime import datetime
from keras.utils import to_categorical
import scipy as sp
import scipy.signal
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


class ml_pipeline:

    def __init__(self):
        self.pipeline = []

    def add(self, function):
        self.pipeline.append(function)

    def predict(self, x):
        for function in self.pipeline:
            x = function(x)
            #print(x.shape)

        return x


class emg_clf:

    def __init__(self):
        self.model = self.prediction_model()

    def prediction_model(self):
        # We can now create our keras model for our simple CNN architecture
        # create model
        #model = Sequential()
        # add model layers
        #model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(5001, 16, 1)))
        #model.add(Conv2D(32, kernel_size=3, activation='relu'))
        #model.add(Flatten())
        # model.add(Dense(256, activation='relu'))
        #model.add(Dense(13, activation='softmax'))

        model = Sequential()
        model.add(Dense(64, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(5, activation='softmax'))

        return model

    def load_model(self, model_name):
        self.model = tensorflow.keras.models.load_model(model_name)






    #def format_cnn_data(self, data_x, x_channels=['EMG' + str(i) for i in range(1, 17)]):
    def format_cnn_data(self, data_x, x_channels=['HR']):
        # This function takes in a list of window dfs and formats the data into a shape keras expects
        # Shape: (n_samples, n_window_samples, n_features, 1)

        window_x = data_x[x_channels].values
        window_x = window_x.reshape(-1, window_x.shape[0], window_x.shape[1], 1)
        return np.asarray(window_x).astype('float32')
        return

#def filter_all_channels(filt_emg, emg_keys=['EMG' + str(i) for i in range(1, 17)]):
def filter_all_channels(filt_emg, emg_keys=['HR']):
    return filt_emg[emg_keys].apply(filteremg)


def filteremg(emg, low_pass=10, sfreq=1000, high_band=20, low_band=450):
    """
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """
    # normalise cut-off frequencies to sampling frequency
    #high_band = high_band / (sfreq / 2)
    #low_band = low_band / (sfreq / 2)

    # create bandpass filter for EMG
    #b1, a1 = sp.signal.butter(4, [high_band, low_band], btype='bandpass')

    # process EMG signal: filter EMG
    #emg_filtered = sp.signal.filtfilt(b1, a1, emg)

    # process EMG signal: rectify
    #emg_rectified = abs(emg_filtered)

    # create lowpass filter and apply to rectified signal to get EMG envelope
    #low_pass = low_pass / (sfreq / 2)
    #b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
    #emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)

    #return emg_envelope
    return


