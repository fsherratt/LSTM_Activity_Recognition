import tensorflow as tf
from tensorflow import keras


class EarlyStoppingCallback(keras.callbacks.Callback):
    def __init__(self, threshold: float, **kwargs):
        """
        Threshold defines the percentage accuracy required before early stopping is called
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Has training exceeded threshold?
        if logs["val_categorical_accuracy"] > self.threshold:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Early Stopping at {} %".format(self.threshold * 100))
