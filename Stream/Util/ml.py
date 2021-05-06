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
        model = Sequential()
        model.add(Dense(64, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(5, activation='softmax'))

        return model

    def load_model(self, model_name):
        self.model = tensorflow.keras.models.load_model(model_name)



    def format_cnn_data(self, data_x, x_channels=['HR']):
        # This function takes in a list of window dfs and formats the data into a shape keras expects
        # Shape: (n_samples, n_window_samples, n_features, 1)

        window_x = data_x[x_channels].values
        window_x = window_x.reshape(-1, window_x.shape[0], window_x.shape[1], 1)
        return np.asarray(window_x).astype('float32')
        return



