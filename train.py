"""
This file setups and run ML training based of provided config files
"""
import argparse
import datetime
import collections
from logging import error
import os
import math
import pathlib
import copy
import random
import re
from typing import Collection

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from sklearn import metrics as skmetrics
from sklearn.utils import class_weight, shuffle

import hparam_load
from earlyStoppingCallback import EarlyStoppingCallback


class InsufficientData(Exception):
    pass


def hardware_setup(use_gpu: True, random_seed: 0):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    if use_gpu:
        try:
            physical_devices = tf.config.experimental.list_physical_devices("GPU")
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

            print("GPU Device")
            tf.test.gpu_device_name()
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
):
    # class_weight_val = np.ones(training_data[1].shape[-1])

    y = np.argmax(training_data[1], axis=-1)
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y), y=y
    )
    # class_weights = dict(zip(np.unique(y), class_weights))

    print(f"Class Weights: {class_weights}")

    return tf_model.fit(
        x=training_data[0],
        y=training_data[1],
        validation_data=validation_data,
        callbacks=callbacks,
        # class_weight=class_weights,
        **settings,
    )


class Data_Files:
    def __init__(
        self, data_folder: str, verbose=True, shuffle=False, preload=True,
    ):
        self.verbose = verbose

        self.extension = ".csv"
        self.file_dict = {}

        self.file_list = self.get_file_list(data_folder, self.extension)

        if shuffle:
            random.seed(1)
            random.shuffle(self.file_list)

        if preload:
            self.preload_data()

    def find_cache_data(self, data_file):
        if data_file not in self.file_dict:
            if self.verbose:
                print(f"Loading {data_file.name}")
            self.file_dict[data_file] = self.load_file(data_file)

        return self.file_dict[data_file]

    def preload_data(self):
        for file_name in self.file_list:
            if self.load_augment or not self.is_augmented_data(str(file_name)):
                self.file_dict[file_name] = self.load_file(file_name)

    def exclude_filter(self, filter_dir: str) -> list:
        if filter_dir is None:
            return self.file_list

        return [
            file_name
            for file_name in self.file_list
            if self.check_validity(str(file_name), filter_dir)
        ]

    def include_filter(self, filter_dir: str) -> list:
        if filter_dir is None:
            return self.file_list

        return [
            file_name
            for file_name in self.file_list
            if not self.check_validity(str(file_name), filter_dir)
        ]

    def _gen_activity_file_lists(self, data_files: list) -> dict:
        """
        Seperate out the different file types into seperate lists

        @param data_files: list
            A list of data file Path objects

        @return dict
            Dictionary of lists - one list for each activity type
        """
        activity_list = collections.defaultdict(list)

        for file in data_files:
            activity_list[self.get_activity_type(file.name)].append(file)

        return dict(activity_list)

    @staticmethod
    def _offset_file_dict(data_files: dict, offset: int) -> dict:
        """
        Offset all of the lists in a dictionary

        @param data_files: dict
            Dictionary of lists - one list for each activity type

        @param offset: int
            Amount of entries to offset the list by

        @return dict
            Dictionary of lists - one list for each activity type
        """
        for key, value in data_files.items():
            if offset > len(value):
                raise InsufficientData(
                    f"The number of data files for '{key}' is less than offset {offset}"
                )

            value = value[offset:] + value[:offset]

        return data_files

    @staticmethod
    def _clip_per_activity_data_windows(data_dict: dict, max_items: int):
        for key, value in data_dict.items():
            perms = np.random.permutation(len(value[1]))
            perms = list(perms[:max_items])

            data = [value[0][i] for i in perms]
            labels = [value[1][i] for i in perms]

            data_dict[key] = (data, labels)

        return data_dict

    def _parse_episode_list(
        self, input_data_files: dict, target_samples: int, parse_settings: dict
    ) -> tuple:
        """
        Parse the episode list until the correct amount of samples has been reached. A IndexError is raised if not enough data available

        @param input_data_files : dict
             Dictionary of lists - one list for each activity type

        @param target_samples: int
            Target number of data samples

        @param parse_settings: dict
            Dictionary of parsing settings
        """
        activity_data_files = copy.copy(input_data_files)

        activity_samples = collections.defaultdict(lambda: ([], []))
        for key, value in activity_data_files.items():
            while len(activity_samples[key][1]) < target_samples:
                try:
                    file = value.pop(0)
                except IndexError:
                    raise InsufficientData(f"Not enough data for activity {key}")

                x, y = self.parse_file(self.find_cache_data(file), **parse_settings)

                activity_samples[key][0].extend(x)
                activity_samples[key][1].extend(y)

        return dict(activity_samples), activity_data_files

    def load_data_episodes(
        self,
        parse_kwargs: dict,
        file_offset: int,
        test_samples_min: int,
        train_samples_min: int,
        valid_samples_min: int,
        filter_mode=True,  # True = exclude items in list, False = only load list
        filter_list=None,  # List
    ) -> dict:
        """
        Parse data files for episode based data

        @param parse_kwargs: dict
            Keyword arguments for the data parser

        @param file_offset: int
            The starting point in the data file list - allows for consistent shuffling of data

        @param test_samples_min: int
            The minimum number of test samples to return

        @param train_samples_min: int
            The minimum number of training samples to return

        @param valid_samples_min: int
            The minimum number of validation samples to return

        @filter mode: bool
            True if items in the filter list directories should be excluded, False if only filter list directories should be included

        @return dict of train, test and validation data dictionaries
        """

        if filter_mode is None:
            data_files = self.file_list
        elif filter_mode:
            data_files = self.exclude_filter(filter_list)
        else:
            data_files = self.include_filter(filter_list)

        activity_file_dict = self._gen_activity_file_lists(data_files)
        activity_file_dict = self._offset_file_dict(activity_file_dict, file_offset)

        # Parse episodes until the correct amount of data is obtained
        # Test data
        test_data, activity_file_dict = self._parse_episode_list(
            activity_file_dict, test_samples_min, parse_kwargs
        )

        # Train data
        train_data, activity_file_dict = self._parse_episode_list(
            activity_file_dict, train_samples_min, parse_kwargs
        )

        # Validataion data
        valid_data, activity_file_dict = self._parse_episode_list(
            activity_file_dict, valid_samples_min, parse_kwargs
        )

        return {"train": train_data, "valid": valid_data, "test": test_data}

    def load_data_activities(
        self,
        parse_kwargs: dict,
        file_offset: int,
        test_samples: int,
        train_samples: int,
        valid_samples: int,
        filter_mode=True,
        filter_list=None,
    ):
        """
        Extends load data episodes by adding left/right ankle, clipping max data, combining activities

        @param parse_kwargs: dict
            Keyword arguments for the data parser

        @param file_offset: int
            The starting point in the data file list - allows for consistent shuffling of data

        @param test_sample: int
            The number of test samples to return

        @param train_sample: int
            The number of training samples to return

        @param valid_sample: int
            The number of validation samples to return

        @filter mode: bool
            True if items in the filter list directories should be excluded, False if only filter list directories should be included

        @return dict of train, test and validation data dictionaries
        """
        data_episodes_kwargs = {
            "parse_kwargs": parse_kwargs,
            "file_offset": file_offset,
            "test_samples_min": int(test_samples / 2),
            "train_samples_min": int(train_samples / 2),
            "valid_samples_min": int(valid_samples / 2),
            "filter_mode": filter_mode,
            "filter_list": filter_list,
        }

        r_parse_kwargs = copy.copy(parse_kwargs)
        r_parse_kwargs["data_headings"] = [
            i for i in parse_kwargs["data_headings"] if i.startswith("r")
        ]

        l_parse_kwargs = copy.copy(parse_kwargs)
        l_parse_kwargs["data_headings"] = [
            i for i in parse_kwargs["data_headings"] if i.startswith("l")
        ]

        # Load data for left and right ankles
        data_episodes_kwargs["parse_kwargs"] = r_parse_kwargs
        r_data = self.load_data_episodes(**data_episodes_kwargs)

        data_episodes_kwargs["parse_kwargs"] = l_parse_kwargs
        l_data = self.load_data_episodes(**data_episodes_kwargs)

        # Clip data
        for key, value in r_data.items():
            r_data[key] = self._clip_per_activity_data_windows(
                value, data_episodes_kwargs[key + "_samples_min"]
            )

        for key, value in l_data.items():
            l_data[key] = self._clip_per_activity_data_windows(
                value, data_episodes_kwargs[key + "_samples_min"]
            )

        # Combine into one data set
        data = activity_samples = collections.defaultdict(lambda: ([], []))

        for key, value in r_data.items():
            for activity_value in value.values():
                data[key][0].extend(activity_value[0])
                data[key][1].extend(activity_value[1])

        for key, value in l_data.items():
            for activity_value in value.values():
                data[key][0].extend(activity_value[0])
                data[key][1].extend(activity_value[1])

        # Convert to np.array
        for key, value in data.items():
            data[key] = (np.asarray(value[0]), value[1])

        return dict(data)

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

        # Identify the data file activity type

        if filter_mode:
            data_files = self.exclude_filter(**filter)
        else:
            data_files = self.include_filter(**filter)

        for file in data_files:

            if self.verbose:
                print("Loaded file {}".format(str(file)))
            new_x, new_y = self.parse_file(self.find_cache_data(file), **parse_settings)

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
    def get_activity_type(file_name: str) -> tuple:
        # Search file name to try and identify activity and split index
        # Filename in format ABC_freq_ACTIVITY_NAME_Log_Date_time.csv
        # returns None is match not found
        # returns (activity string, date, file, split index) if match found
        re_output = re.search(r"[\w]*_\d*_([\w]*)_Log_(\d*)_(\d*)_(\d*).csv", file_name)

        if not re_output:
            return None

        file_activity = re_output.group(1)
        file_date = re_output.group(2)
        file_time = re_output.group(3)
        file_seq = re_output.group(4)

        return file_activity

    @staticmethod
    def check_validity(file_name: str, exclude_dirs) -> bool:
        return all(exclude not in file_name for exclude in exclude_dirs)

    @staticmethod
    def get_file_list(data_directory: str, extension: str) -> list:
        data_directory = pathlib.Path(data_directory)
        files = os.listdir(data_directory)
        data_files = []

        for file in files:
            file = data_directory / file

            if os.path.isdir(file):
                data_files.extend(Data_Files.get_file_list(file, extension))

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


def label_to_one_hot(labels, num_activities, label_mapping=None):
    if label_mapping is not None:
        labels = list(map(lambda x: label_mapping[x], labels))

    labels = tf.one_hot(labels, num_activities)

    return np.asarray(labels, dtype=np.int8)


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


def gen_conf_matrix(data, labels, data_type: str):
    actual_class = tf.math.argmax(labels, axis=-1)
    predicted_class = tf.math.argmax(model.predict(data), axis=-1)

    acc = sum(np.asarray(actual_class) == np.asarray(predicted_class)) / len(
        actual_class
    )
    print(f"*** {data_type} Performance ***")

    print(f"{data_type} values {tf.math.reduce_sum(labels.astype(np.int32), axis=0)}")
    print(f"Accuracy: {acc:.3f}%")
    conf_matrix = tf.math.confusion_matrix(
        labels=actual_class,
        predictions=predicted_class,
        num_classes=conf["data"]["data_settings"]["num_labels"],
    )
    print(conf_matrix)

    return actual_class, predicted_class


def save_hparam_to_file(file, header, values):
    with open(pathlib.Path(file), mode="a") as file:
        file.write(",".join(map(str, header)) + "\n")
        file.write(",".join(map(str, values)) + "\n")


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

    data_files = Data_Files(
        data_folder=_conf["data"]["folder"],
        verbose=_conf["data"]["verbose"],
        shuffle=True,
        preload=False,
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

            # Print out hyperparamaters
            print("\n------------------------")
            print(f"Start: {start_time}")
            print(f"Sweep: {ix + 1} of {len(hparams)}")
            print(f"Hyperparamaters: {hparam_set}")
            print("------------------------\n")

            # Import and process data
            # Filter out participant for cross validation study
            try:
                train_samples = int(
                    conf["data"]["training_samples"] * conf["data"]["test_train_split"]
                )
                valid_samples = int(
                    conf["data"]["training_samples"]
                    * (1 - conf["data"]["test_train_split"])
                )
                data = data_files.load_data_activities(
                    parse_kwargs=conf["data"]["data_settings"],
                    file_offset=conf["data"]["episode_offset"],
                    train_samples=train_samples,
                    valid_samples=valid_samples,
                    test_samples=conf["data"]["test_samples"],
                    filter_mode=False,  # Only Include participants in list
                    filter_list=conf["data"]["x_validation_exclude"],
                )

                train_data = data["train"]
                validation_data = data["valid"]
                test_data = data["test"]

            except InsufficientData as exc:
                print(exc)
                continue

            print("Training Data Samples: {}".format(len(train_data[1])))
            print("Validation Data Samples: {}".format(len(validation_data[1])))
            print("Test Data Samples: {}".format(len(test_data[1])))

            # Convert to one hot
            # Train Data
            train_data_labels = label_to_one_hot(
                train_data[1],
                conf["data"]["data_settings"]["num_labels"],
                conf["data"]["data_settings"]["label_mapping"],
            )
            train_data = (train_data[0], train_data_labels)

            # Validation Data
            validation_data_labels = label_to_one_hot(
                validation_data[1],
                conf["data"]["data_settings"]["num_labels"],
                conf["data"]["data_settings"]["label_mapping"],
            )
            validation_data = (validation_data[0], validation_data_labels)

            # Test data
            test_data_labels = label_to_one_hot(
                test_data[1],
                conf["data"]["data_settings"]["num_labels"],
                conf["data"]["data_settings"]["label_mapping"],
            )
            test_data = (test_data[0], test_data_labels)

            # ------------------------------------------------------------------------------
            # Set up ML model
            input_shape = train_data[0].shape[-2:]
            print("Input Shape: {}".format(input_shape))
            model = create_model(
                layer_definitions=model_conf["layers"], input_shape=input_shape
            )

            loss_func = loss_function(
                conf["loss_func"]["type"], conf["loss_func"]["settings"]
            )
            model = compile_model(
                tf_model=model, loss_func=loss_func, settings=conf["compile"]
            )
            model_save_dir = pathlib.Path(
                conf["save"]["model_dir"] + "/" + start_time + "/"
            )

            # ------------------------------------------------------------------------------
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

            if conf["callbacks"]["use_early_stopping_threshold"]:
                callback_list.append(
                    early_stopping_threshold_callback(
                        stop_threshold=conf["callbacks"]["early_stopping_threshold"]
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

            # ------------------------------------------------------------------------------
            # Print list of classes
            print(
                "Using callbacks: {}".format(
                    ", ".join(x.__class__.__name__ for x in callback_list)
                )
            )

            if conf["save"]["config"]:
                save_folder = conf["save"]["config_dir"] + start_time

                save_copy_config(save_folder, "config.yaml", conf)
                save_copy_config(
                    save_folder, "model.yaml", model_conf,
                )

            # ------------------------------------------------------------------------------
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

            # ------------------------------------------------------------------------------
            # Print data summary
            gen_conf_matrix(train_data[0], train_data[1], "train")
            gen_conf_matrix(validation_data[0], validation_data[1], "validation")
            actual_class, predicted_class = gen_conf_matrix(
                test_data[0], test_data[1], "test"
            )

            # ------------------------------------------------------------------------------
            # Save results
            if conf["hyper_paramaters"]["save_hparam"]:
                # Analyse performance
                class_report = analyse_test_data(
                    y_true=actual_class,
                    y_pred=predicted_class,
                    num_classes=conf["data"]["data_settings"]["num_labels"],
                )

                # Save results to log files
                # Make participant exclusion data CSV friendly
                hparam_set["HP_VALIDATION_INCLUDE"] = "-".join(
                    map(str, hparam_set["HP_VALIDATION_INCLUDE"])
                )

                header = [
                    "Timestamp",
                    "Sweep",
                    "Sweep Total",
                    *hparam_set.keys(),
                ]
                values = [start_time, ix + 1, len(hparams), *hparam_set.values()]
                save_hparam_to_file(
                    conf["hyper_paramaters"]["hparam_log_file"], header, values
                )

                # Results
                header = [
                    "Timestamp",
                    "params",
                    "epochs",
                    "train_accuracy",
                    "val_accuracy",
                    *class_report.keys(),
                ]

                values = [
                    start_time,
                    model.count_params(),
                    history.epoch[-1] + 1,
                    history.history["categorical_accuracy"][-1],
                    history.history["val_categorical_accuracy"][-1],
                    *class_report.values(),
                ]
                save_hparam_to_file(
                    conf["hyper_paramaters"]["hparam_result_file"], header, values
                )

        except KeyboardInterrupt:
            break
