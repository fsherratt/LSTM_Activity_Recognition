# UCI Human Activity Database Classification
# https://towardsdatascience.com/how-to-quickly-build-a-tensorflow-training-pipeline-15e9ae4d78a0
# https://www.tensorflow.org/beta/tutorials/text/time_series

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

tf.random.set_seed(13)

mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['axes.grid'] = False

# Create tensorflow model of LSTM
class Model(object):
    def __init__(self, test_generator, valid_generator):
        self.tGenerator = test_generator
        self.vGenerator = valid_generator
        self.model = None

    def create_model(self, hidden_size=50, use_dropout=True):
        self.model = keras.models.Sequential()

        # Return sequences returns output from each LSTM cell
        self.model.add(keras.layers.LSTM(hidden_size, return_sequences=True))

        if use_dropout:
            self.model.add(keras.layers.Dropout(0.2))

        self.model.add(keras.layers.Activation("softmax"))

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam() )

    def fit_generator(self):

        data_size = 10
        num_epochs = 5
        self.model.fit_generator(self.tGenerator, epochs=num_epochs, steps_per_epoch=data_size)


# Create generator that produces a window of data from the pandas object
class WindowGenerator():
    def __init__(self, filename):
        self._filename = filename
        self._labelHeading = 'Activity'

        self._windowSize = 50
        self._shift = 1
        self._stride = 1

        self.dataframe = pd.read_csv(self._filename)
        self.label = self.dataframe.pop(self._labelHeading)

        self._num_activities = 10

    def get_next_window(self):

        for i in range(len(self.dataframe) - self._windowSize):
            # Extract values for window
            x = self.dataframe[i:i+self._windowSize].values
            y = self.label[i+self._windowSize]

            # Convert y to one hot
            y = tf.one_hot(y, self._num_activities)

            yield((x, y))

def main():
    test_data_path = 'UCI_HAD/Transform_Data/data_exp001.csv'
    valid_data_path = 'UCI_HAD/Transform_Data/data_exp002.csv'

    test_generator = WindowGenerator(test_data_path).get_next_window()
    valid_generator = WindowGenerator(valid_data_path).get_next_window()

    model = Model(test_generator=test_generator, valid_generator=valid_generator)
    model.create_model()
    model.compile_model()
    model.fit_generator()


if __name__ == '__main__':
    main()
