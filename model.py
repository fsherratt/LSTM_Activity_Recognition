import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics as skmetrics
from sklearn.utils import class_weight
from tensorflow import keras


class EarlyStoppingCallback(keras.callbacks.Callback):
    def __init__(self, stop_threshold: float, **kwargs):
        """
        Threshold defines the percentage accuracy required before early stopping is called
        """
        super().__init__(**kwargs)
        self.stop_threshold = stop_threshold
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Has training exceeded threshold?
        if logs["val_categorical_accuracy"] > self.stop_threshold:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Early Stopping at {} %".format(self.stop_threshold * 100))


def create_model(layer_definitions: list, input_shape: list) -> tf.keras.Sequential:
    # Add in layer type definitions here
    layer_dict = {
        "lstm": tf.keras.layers.LSTM,
        "lstm_peephole": tf.keras.experimental.PeepholeLSTMCell,
        "bidirectional": tf.keras.layers.Bidirectional,
        "dense": tf.keras.layers.Dense,
        "activation": tf.keras.layers.Activation,
        "dropout": tf.keras.layers.Dropout,
        "flatten": tf.keras.layers.Flatten,
        "conv1d": tf.keras.layers.Conv1D,
        "conv2d": tf.keras.layers.Conv2D,
        "maxpool1D": tf.keras.layers.MaxPooling1D,
        "avgpool1D": tf.keras.layers.AvgPool1D,
    }

    activation_dict = {
        "relu": tf.nn.relu,
        "leaky_relu": tf.nn.leaky_relu,
        "softmax": tf.nn.softmax,
    }

    # Add input definition to first layer
    layer_definitions[0]["args"]["input_shape"] = input_shape

    tf_model = tf.keras.Sequential()

    for layer in layer_definitions:
        if layer["enabled"]:
            if layer["type"] == "bidirectional":
                layer["args"]["layer"] = layer_dict[layer["args"]["layer"]["type"]](
                    **layer["args"]["layer"]["args"]
                )

            if layer["args"] is None:
                layer["args"] = {}

            if layer["type"] == "activation":
                layer["args"]["activation"] = activation_dict[
                    layer["args"]["activation"]
                ]

            tf_model.add(layer_dict[layer["type"]](**layer["args"]))

    return tf_model


def loss_function(loss_type, settings) -> tf.keras.losses.Loss:
    loss_funcs = {"categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy}
    return loss_funcs[loss_type](**settings)


def compile_model(
    tf_model: tf.keras.Sequential, settings: dict, loss_func: tf.keras.losses.Loss
) -> tf.keras.Sequential:
    tf_model.compile(loss=loss_func, **settings)
    tf_model.summary()

    return tf_model


def fit_model(
    tf_model: tf.keras.Sequential,
    training_data,
    validation_data,
    callbacks: None,
    settings=None,
    use_class_weights=False,
):

    if settings is None:
        settings = {}

    if use_class_weights:
        # class_weight_val = np.ones(training_data[1].shape[-1])

        y = np.argmax(training_data[1], axis=-1)
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
        settings["class_weights"] = class_weights
        print(f"Class Weights: {class_weights}")
        # class_weights = dict(zip(np.unique(y), class_weights))

    return tf_model.fit(
        x=training_data[0],
        y=training_data[1],
        validation_data=validation_data,
        callbacks=callbacks,
        **settings,
    )


def tensorboard_callback(settings, save_dir):
    return tf.keras.callbacks.TensorBoard(log_dir=save_dir, **settings)


def early_stopping_threshold_callback(stop_threshold):
    # Custom callback to stop training after validation threshold reached
    return EarlyStoppingCallback(stop_threshold=stop_threshold)


def early_stopping_callback(settings):
    return tf.keras.callbacks.EarlyStopping(**settings)


def save_model_callback(settings, save_path):
    save_path = save_path / "cp-{epoch:04d}.ckpt"
    return tf.keras.callbacks.ModelCheckpoint(filepath=save_path.__str__(), **settings)


def label_to_one_hot(labels, num_activities, label_mapping=None):
    if label_mapping is not None:
        labels = list(map(lambda x: label_mapping[x], labels))

    labels = tf.one_hot(labels, num_activities)

    return np.asarray(labels, dtype=np.int8)


def analyse_test_data(y_true, y_pred, num_classes) -> dict:
    default_keys = [
        "accuracy",
        "micro avg_precision",
        "micro avg_recall",
        "micro avg_f1-score",
        "micro avg_support",
        "macro avg_precision",
        "macro avg_recall",
        "macro avg_f1-score",
        "macro avg_support",
        "weighted avg_precision",
        "weighted avg_recall",
        "weighted avg_f1-score",
        "weighted avg_support",
    ]
    target_names = ["W", "RA", "RD", "SA", "SD", "S"]

    # Analyse performance
    class_report = skmetrics.classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names,
        labels=np.arange(num_classes),
        output_dict=True,
        zero_division=0,
    )

    class_report = pd.io.json._normalize.nested_to_record(class_report, sep="_")

    class_report["accuracy"] = skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    # Merge with default headers and return
    default_value = -1
    default_result_dict = dict.fromkeys(default_keys, default_value)
    return {**default_result_dict, **class_report}


def gen_conf_matrix(model, data, labels, num_labels=None):
    """
    @param model: tf.model
        Tensorflow model to test

    @param data: np.array
        Data windows to test model against
    
    @param labels: np.array
        Corresponding data labels
    """
    if num_labels is None:
        num_labels = len(np.unique(labels))

    actual_class = tf.math.argmax(labels, axis=-1)
    predicted_class = tf.math.argmax(model.predict(data), axis=-1)

    acc = sum(np.asarray(actual_class) == np.asarray(predicted_class)) / len(
        actual_class
    )

    print(f"Values {tf.math.reduce_sum(labels.astype(np.int32), axis=0)}")
    print(f"Accuracy: {acc:.3f}%")
    conf_matrix = tf.math.confusion_matrix(
        labels=actual_class, predictions=predicted_class, num_classes=num_labels,
    )
    print(conf_matrix)

    return actual_class, predicted_class

