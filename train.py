"""
This file setups and run ML training based of provided config files
"""
import argparse
import datetime
import os
import math
import pathlib
import shutil
import copy
import re

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from sklearn import metrics as skmetrics
from sklearn.utils import class_weight

import hparam_load


def hardware_setup(use_gpu: True, random_seed: 0):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    if use_gpu:
        try:
            physical_devices = tf.config.experimental.list_physical_devices("GPU")
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_cli():
    parser = argparse.ArgumentParser(description="Startup module(s)")
    parser.add_argument(
        "-c", "-C", "--config-file", type=str, default=None, help="Config file path",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--percent-traning-data",
        type=float,
        default=None,
        help="Percentage Training Data",
    )
    return parser.parse_args()


def load_config(config_file) -> dict:
    with open(config_file) as file:
        config = yaml.full_load(file)

    return config


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
            tf_model.add(layer_dict[layer["type"]](**layer["args"]))

    return tf_model


def loss_function(type, settings) -> tf.keras.losses.Loss:
    loss_funcs = {"categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy}
    return loss_funcs[type](**settings)


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
):
    class_weights = np.ones(training_data[1].shape[-1])

    y = np.argmax(training_data[1], axis=-1)
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y), y=y
    )

    print("Class Weights: ", end="")
    print(class_weights)

    history = tf_model.fit(
        x=training_data[0],
        y=training_data[1],
        validation_data=validation_data,
        callbacks=callbacks,
        class_weight=class_weights,
        **settings,
    )

    return history


class Data_Preload:
    def __init__(self, data_folder: str, load_augment=True, verbose=True):
        self.load_augment = load_augment

        self.verbose = verbose

        self.file_list = []
        self.extension = ".csv"
        self.file_dict = dict()

        self.file_list = self.get_file_list(data_folder, self.extension)
        self.preload_data()

    def preload_data(self):
        for file_name in self.file_list:
            if self.load_augment or not self.is_augmented_data(str(file_name)):
                if self.verbose:
                    print("Loading {}".format(file_name))
                self.file_dict[file_name] = self.load_file(file_name)

    def exclude_filter(
        self, filter_dir: str, include_aug: bool, freq_min: int, freq_max: int
    ) -> list:
        if filter_dir is None:
            return self.file_dict.keys()

        return_list = []

        for file_name in self.file_dict.keys():
            if not self.check_validity(str(file_name), filter_dir):
                continue

            if (
                not include_aug
                and self.is_augmented_data(str(file_name)) is not include_aug
            ):
                continue

            if not self.check_freq(str(file_name), freq_min, freq_max):
                continue

            return_list.append(file_name)

        return return_list

    def include_filter(
        self, filter_dir: str, include_aug: bool, freq_min: int, freq_max: int
    ) -> list:
        if filter_dir is None:
            return self.file_dict.keys()

        return_list = []

        for file_name in self.file_dict.keys():
            if self.check_validity(str(file_name), filter_dir):
                continue

            if self.is_augmented_data(str(file_name)) is not include_aug:
                continue

            if not self.check_freq(str(file_name), freq_min, freq_max):
                continue

            return_list.append(file_name)

        return return_list

    def load_data(
        self,
        parse_settings: dict,
        filter_mode=True,  # True = exclude items in list, False = only load list
        filter_list=None,  # List
        include_aug=True,
        freq_min=None,  # Int
        freq_max=None,  # Int
        append=False,
    ):
        x = []
        y = []

        data_files = []

        filter = {
            "filter_dir": filter_list,
            "include_aug": include_aug,
            "freq_min": freq_min,
            "freq_max": freq_max,
        }

        if filter_mode:
            data_files = self.exclude_filter(**filter)
        else:
            data_files = self.include_filter(**filter)

        for file in data_files:
            if self.verbose:
                print("Loaded file {}".format(str(file)))
            new_x, new_y = self.parse_file(self.file_dict[file], **parse_settings)

            if append:
                x.append(new_x)
                y.append(new_y)
            else:
                x.extend(new_x)
                y.extend(new_y)

        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    @staticmethod
    def check_freq(file_name: str, freq_min: int, freq_max: int) -> bool:
        if freq_min is None:
            freq_min = -math.inf

        if freq_max is None:
            freq_max = math.inf

        freq = re.search(r"(?:Aug|Out)_(\d*)", file_name).group(1)
        freq = int(freq)

        if freq_min <= freq <= freq_max:
            return True

        return False

    @staticmethod
    def is_augmented_data(file_name: str) -> bool:
        if "Aug_" in file_name:
            return True

        return False

    @staticmethod
    def check_validity(file_name: str, exclude_dirs) -> bool:
        for exclude in exclude_dirs:
            if exclude in file_name:
                return False

        return True

    @staticmethod
    def get_file_list(data_directory: str, extension: str) -> list:
        data_directory = pathlib.Path(data_directory)
        files = os.listdir(data_directory)
        data_files = []

        for file in files:
            file = data_directory / file

            if os.path.isdir(file):
                data_files.extend(Data_Preload.get_file_list(file, extension))

            elif file.exists() and file.suffix == extension:
                data_files.append(file)

        return data_files

    @staticmethod
    def load_file(file) -> pd.core.frame.DataFrame:
        return pd.read_csv(file)

    @staticmethod
    def parse_file(
        data,
        label_heading,
        data_headings,
        num_timesteps,
        num_labels,
        skip,
        normalize,
        label_mapping=None,
    ):
        labels_data = data[label_heading]
        labels_data = np.asarray(labels_data, dtype=np.int32)
        input_data = data[data_headings]
        input_data = np.asarray(input_data, dtype=np.float32)

        # ----------------------------------------------------------------------------------
        # Output filter and mapping
        if label_mapping is not None:
            included_activites = np.asarray(list(label_mapping.keys()))

            # Filter
            filter_list = np.isin(labels_data, included_activites)
            labels_data = labels_data[filter_list]
            input_data = input_data[filter_list]

            # Map data based on dict
            # labels_data = list(map(lambda x: label_mapping[x], labels_data))
        # ----------------------------------------------------------------------------------

        # Apply one_hot to labels
        # labels_data = tf.one_hot(labels_data, num_labels)
        labels_data = np.asarray(labels_data, dtype=np.int8)
        input_data = np.asarray(input_data, dtype=np.float32)

        data_frames = []
        label_frames = []

        i = 0
        max_start_index = labels_data.shape[0] - num_timesteps

        while i < max_start_index:
            sample_end = i + num_timesteps

            data_frames.append(input_data[i:sample_end, :])
            label_frames.append(labels_data[sample_end])

            i += skip + 1

        # Normalise for each data window
        if normalize:
            data_frames = tf.keras.utils.normalize(data_frames, axis=1, order=2)

        return data_frames, label_frames


# Percentage train allows
def split_test_train(data, labels, split, percent_train=1.0, max_difference=1.5):
    train_rows = []
    test_rows = []

    # Do something to limit max difference between labels
    unique_labels, label_count = np.unique(labels, return_counts=True)
    # Ensure that we have some data to train with
    new_label_count = label_count[label_count > 0.5 * np.median(label_count)]
    if len(label_count) != len(new_label_count):
        print("!!!!!Quantiy of labels is too varied!!!!!")

    min_labels = np.min(new_label_count)

    for i in range(unique_labels.shape[0]):
        # Limit to 50% difference
        if max_difference is not None:
            if label_count[i] > max_difference * min_labels:
                label_count[i] = max_difference * min_labels

        train_test_split = int(split * label_count[i])

        label_rows = np.where(labels == unique_labels[i])[0]
        perms = np.random.permutation(label_count[i])

        train_rows.extend(label_rows[perms[: int(train_test_split * percent_train)]])
        test_rows.extend(label_rows[perms[train_test_split:]])

    print(label_count)
    train = (
        data.take(train_rows, axis=0),
        labels.take(train_rows, axis=0),
    )
    test = (
        data.take(test_rows, axis=0),
        labels.take(test_rows, axis=0),
    )

    return test, train


def label_to_one_hot(labels, num_activities, label_mapping=None):
    if label_mapping is not None:
        labels = list(map(lambda x: label_mapping[x], labels))

    labels = tf.one_hot(labels, num_activities)

    return np.asarray(labels, dtype=np.int8)


def tensorboard_callback(settings, save_dir):
    return tf.keras.callbacks.TensorBoard(log_dir=save_dir, **settings)


def early_stopping_callback(settings):
    return tf.keras.callbacks.EarlyStopping(**settings)


def save_model_callback(settings, save_path):
    save_path = save_path / "cp-{epoch:04d}.ckpt"
    return tf.keras.callbacks.ModelCheckpoint(filepath=save_path.__str__(), **settings)


def save_copy_config(file_dir: str, config_file: str, config: dict):
    config_file = pathlib.Path(config_file)
    file_dir = pathlib.Path(file_dir)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    file_path = file_dir / config_file.name
    with open(file_path, "w") as file:
        file.write(yaml.dump(config))


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


if __name__ == "__main__":
    args = parse_cli()

    # Load configuration files
    _conf = load_config(config_file=args.config_file)
    _model_conf = load_config(config_file=_conf["model"]["config_file"])
    hparams = hparam_load.load_hparam_set(_conf["hyper_paramaters"]["hparam_file"])

    if args.seed:
        _conf["hardware_setup"]["random_seed"] = args.seed

    if args.percent_traning_data:
        _conf["data"]["percentage_train"] = args.percent_traning_data

    # Setup physical devices
    hardware_setup(**_conf["hardware_setup"])

    data_files = Data_Preload(
        data_folder=_conf["data"]["folder"],
        load_augment=_conf["data"]["load_augment"],
        verbose=_conf["data"]["verbose"],
    )

    # Hyper paramater training
    start_ix_offset = 0
    for ix, hparam_set in enumerate(hparams):
        if ix + 1 < start_ix_offset:
            continue
        try:
            start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            # Update config with next hyperparamater set
            conf = copy.deepcopy(_conf)
            conf = hparam_load.set_hparams(hparam_set, conf)

            model_conf = copy.deepcopy(_model_conf)
            model_conf = hparam_load.set_hparams(hparam_set, model_conf)

            # Save copy of hyperparamaters
            print("\n------------------------")
            print("Start: {}".format(start_time))
            print("Sweep: {} of {}".format(ix + 1, len(hparams)))
            print("Hyperparamaters: ", end="")
            print(hparam_set)
            print("------------------------\n")

            # Import and process data
            # Filter out participant for cross validation study
            raw_data, label_data = data_files.load_data(
                filter_mode=True,
                filter_list=conf["data"]["x_validation_exclude"],
                include_aug=False,
                parse_settings=conf["data"]["data_settings"],
            )

            validation_data, train_data = split_test_train(
                raw_data,
                label_data,
                split=conf["data"]["test_train_split"],
                percent_train=conf["data"]["percentage_train"],
                max_difference=conf["data"]["max_label_difference"],
            )

            # Load test data
            # Load filtered participants
            test_data, test_label_data = data_files.load_data(
                filter_mode=False,
                filter_list=conf["data"]["x_validation_exclude"],
                include_aug=False,
                parse_settings=conf["data"]["data_settings"],
            )

            # Validation Data
            validation_data_labels = label_to_one_hot(
                validation_data[1],
                conf["data"]["data_settings"]["num_labels"],
                conf["data"]["data_settings"]["label_mapping"],
            )
            validation_data = (validation_data[0], validation_data_labels)

            # Test Data
            train_data_labels = label_to_one_hot(
                train_data[1],
                conf["data"]["data_settings"]["num_labels"],
                conf["data"]["data_settings"]["label_mapping"],
            )
            train_data = (train_data[0], train_data_labels)

            # Test Data
            _, test_data = split_test_train(
                test_data,
                test_label_data,
                split=1,
                percent_train=conf["data"]["percentage_train"],
                max_difference=conf["data"]["max_label_difference"],
            )

            test_label_data = label_to_one_hot(
                test_data[1],
                conf["data"]["data_settings"]["num_labels"],
                conf["data"]["data_settings"]["label_mapping"],
            )
            test_data = test_data[0]

            # train_data = (np.asarray(raw_data), np.asarray(label_data))
            # validation_data = (np.asarray(test_data), np.asarray(test_label_data))

            # Set up ML model
            input_shape = train_data[0].shape[-2:]
            print("Input Shape: {}".format(input_shape))
            model = create_model(
                layer_definitions=model_conf["layers"], input_shape=input_shape
            )

            #
            dense = model.layers[1]
            print(dense.get_weights())

            loss_func = loss_function(
                conf["loss_func"]["type"], conf["loss_func"]["settings"]
            )
            model = compile_model(
                tf_model=model, loss_func=loss_func, settings=conf["compile"]
            )
            model_save_dir = pathlib.Path(
                conf["save"]["model_dir"] + "/" + start_time + "/"
            )

            # Setup callbacks
            callback_list = []

            if conf["callbacks"]["use_tensorboard"]:
                tensorboard_dir = pathlib.Path(
                    conf["save"]["tensorboard_dir"] + "/" + start_time
                )
                callback_list.append(
                    tensorboard_callback(
                        settings=conf["callbacks"]["tensorboard"],
                        save_dir=tensorboard_dir,
                    )
                )

            if conf["callbacks"]["use_early_stopping"]:
                callback_list.append(
                    early_stopping_callback(
                        settings=conf["callbacks"]["early_stopping"]
                    )
                )

            if conf["callbacks"]["use_save_model"]:
                callback_list.append(
                    save_model_callback(
                        settings=conf["callbacks"]["save_model"],
                        save_path=model_save_dir,
                    )
                )

            if conf["save"]["config"]:
                save_folder = conf["save"]["config_dir"] + start_time

                save_copy_config(save_folder, "config.yaml", conf)
                save_copy_config(
                    save_folder, "model.yaml", model_conf,
                )

            # Run training/learning algorithim
            history = fit_model(
                model, train_data, validation_data, callback_list, conf["fit"]
            )

            # Save final model
            if conf["save"]["final_model"]:
                # Save final model and model properties
                model.save(model_save_dir.__str__())

                # Save model history
                history_save_dir = pathlib.Path(
                    conf["save"]["history_dir"] + "/" + start_time + ".csv"
                )
                pd.DataFrame.from_dict(history.history).to_csv(
                    history_save_dir.__str__(), index=False
                )

            # Print data summary
            print(
                "Class values {}".format(
                    tf.math.reduce_sum(label_data.astype(np.int32), axis=0)
                )
            )
            print(
                "Train values {}".format(
                    tf.math.reduce_sum(train_data[1].astype(np.int32), axis=0)
                )
            )
            print(
                "Val values {}".format(
                    tf.math.reduce_sum(validation_data[1].astype(np.int32), axis=0)
                )
            )
            print(
                "Test values {}".format(
                    tf.math.reduce_sum(test_label_data.astype(np.int32), axis=0)
                )
            )

            test_data = np.asarray(test_data)
            test_label_data = np.asarray(test_label_data)

            # Train - Confusion Matrix
            actual_class = tf.math.argmax(validation_data[1], axis=-1)
            predicted_class = tf.math.argmax(model.predict(validation_data[0]), axis=-1)

            print("*** Validation Confusion Matrix ***")
            conf_matrix = tf.math.confusion_matrix(
                labels=actual_class,
                predictions=predicted_class,
                num_classes=conf["data"]["data_settings"]["num_labels"],
            )
            print(conf_matrix)

            # Test - Confusion matrix
            actual_class = tf.math.argmax(test_label_data, axis=-1)
            pred = model.predict(test_data)
            predicted_class = tf.math.argmax(pred, axis=-1)

            print("*** Test Confusion Matrix ***")
            conf_matrix = tf.math.confusion_matrix(
                labels=actual_class,
                predictions=predicted_class,
                num_classes=conf["data"]["data_settings"]["num_labels"],
            )
            print(conf_matrix)

            if conf["hyper_paramaters"]["save_hparam"]:
                # Analyse performance
                class_report = analyse_test_data(
                    y_true=actual_class,
                    y_pred=predicted_class,
                    num_classes=conf["data"]["data_settings"]["num_labels"],
                )

                # Save results to log files
                # Make participant exclusion data CSV friendly
                hparam_set["HP_X_VALIDATION_EXCLUDE"] = "-".join(
                    map(str, hparam_set["HP_X_VALIDATION_EXCLUDE"])
                )

                header = ["Timestamp", "Sweep", "Sweep Total"]
                header.extend(hparam_set.keys())

                values = [start_time, ix + 1, len(hparams)]
                values.extend(hparam_set.values())
                with open(
                    pathlib.Path(conf["hyper_paramaters"]["hparam_log_file"]), mode="a+"
                ) as file:
                    print(
                        ",".join(map(str, header)), file=file,
                    )
                    print(
                        ",".join(map(str, values)), file=file,
                    )

                # Results
                header = [
                    "Timestamp",
                    "params",
                    "epochs",
                    "train_accuracy",
                    "val_accuracy",
                ]
                header.extend(class_report.keys())

                values = [
                    start_time,
                    model.count_params(),
                    history.epoch[-1] + 1,
                    history.history["categorical_accuracy"][-1],
                    history.history["val_categorical_accuracy"][-1],
                ]
                values.extend(class_report.values())

                with open(
                    pathlib.Path(conf["hyper_paramaters"]["hparam_result_file"]),
                    mode="a+",
                ) as file:
                    print(
                        ",".join(map(str, header)), file=file,
                    )
                    print(
                        ",".join(map(str, values)), file=file,
                    )

        # Catch any errors in code
        except KeyboardInterrupt:
            break

