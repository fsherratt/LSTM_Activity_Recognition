"""
This file setups and run ML training based of provided config files
"""
import argparse
import datetime
import os
import pathlib
import shutil
import copy
import csv

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from sklearn import metrics as skmetrics
from sklearn.utils import class_weight

import hparam_load


def hardware_setup(use_gpu: True, random_seed: 0):
    np.random.seed(random_seed)

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
        "dense": tf.keras.layers.Dense,
        "activation": tf.keras.layers.Activation,
        "dropout": tf.keras.layers.Dropout,
        "reshape": tf.keras.layers.Reshape,
    }

    # Add input definition to first layer
    layer_definitions[0]["args"]["input_shape"] = input_shape

    tf_model = tf.keras.Sequential()

    for layer in layer_definitions:
        if not layer["disabled"]:

            # Deal with the fact that we tensorflow must have a final dimension
            if layer["type"] == "reshape":
                if layer["args"]["target_shape"][0] == None:
                    layer["args"]["target_shape"][0] = np.prod(
                        tf_model.layers[-1].output.shape[1:]
                    )

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


def get_file_list(data_directory: str, exclude_dir=None) -> list:
    data_directory = pathlib.Path(data_directory)
    files = os.listdir(data_directory)
    data_files = []

    for file in files:
        if file == exclude_dir:
            continue

        file = data_directory / file

        if os.path.isdir(file):
            data_files.extend(get_file_list(file))

        elif file.exists() and file.suffix == ".csv":
            data_files.append(file)

    return data_files


def load_data(data_files: list, settings: dict, append=False):
    x = []
    y = []

    i = 1
    i_max = len(data_files)

    for file in data_files:
        print(" {} of {} - Opening {} ".format(i, i_max, file), end="\r")
        i += 1
        # TODO: parameterise constants
        new_x, new_y = parse_file(file, **settings)
        if append:
            x.append(new_x)
            y.append(new_y)
        else:
            x.extend(new_x)
            y.extend(new_y)

    return x, y


def parse_file(
    filename,
    label_heading,
    data_headings,
    num_timesteps,
    num_labels,
    skip,
    normalize=True,
):
    # Read csv file
    data = pd.read_csv(filename)

    # Apply one_hot to labels
    label = data.pop(label_heading)
    label = tf.compat.v2.one_hot(label, num_labels)
    label = np.asarray(label)

    data = data[data_headings]

    # Divide data up into timestep chunks - offset by skip steps
    data_frames = []
    label_frames = []

    i = 0
    max_start_index = data.shape[0] - num_timesteps

    while i < max_start_index:
        sample_end = i + num_timesteps

        data_frames.append(data.values[i:sample_end, :])
        label_frames.append(label[sample_end])

        i += skip + 1

    # Normalise for each data window
    if normalize:
        data_frames = tf.keras.utils.normalize(data_frames, axis=1, order=2)

    return data_frames, label_frames


def split_test_train(data, labels, split, percent_train=1.0):
    data = np.asarray(data)
    labels = np.asarray(labels)

    samples_available = len(labels)
    train_test_split = int(split * samples_available)

    perms = np.random.permutation(data.shape[0])

    train_perms = np.random.permutation(int(train_test_split * percent_train))
    train_perms = perms[train_perms]

    train = (
        data.take(perms[train_perms], axis=0),
        labels.take(perms[train_perms], axis=0),
    )
    test = (
        data.take(perms[train_test_split:], axis=0),
        labels.take(perms[train_test_split:], axis=0),
    )

    return test, train


def tensorboard_callback(settings, save_dir):
    return tf.keras.callbacks.TensorBoard(log_dir=save_dir, **settings)


def early_stopping_callback(settings):
    return tf.keras.callbacks.EarlyStopping(**settings)


def save_model_callback(settings, save_path):
    save_path = save_path / "cp-{epoch:04d}.ckpt"
    return tf.keras.callbacks.ModelCheckpoint(filepath=save_path.__str__(), **settings)


def save_copy_config(file_dir: str, config_file: str):
    config_file = pathlib.Path(config_file)
    file_dir = pathlib.Path(file_dir)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    file_dir = file_dir / config_file.name
    shutil.copy2(config_file.absolute(), file_dir.absolute())


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

            with open(
                pathlib.Path(conf["hyper_paramaters"]["hparam_log_file"]), mode="a+"
            ) as file:
                print(
                    ["Timestamp", "Sweep", "Sweep Total"] + list(hparam_set.keys()),
                    file=file,
                )
                print(
                    [start_time, ix + 1, len(hparams)] + list(hparam_set.values()),
                    file=file,
                )

            # Import and process data
            # Filter out participant for cross validation study
            data_files = get_file_list(
                conf["data"]["folder"], conf["data"]["x_validation_exclude"]
            )
            raw_data, label_data = load_data(data_files, conf["data"]["data_settings"])

            print("Class values {}".format(tf.math.reduce_sum(label_data, axis=0)))

            validation_data, train_data = split_test_train(
                raw_data,
                label_data,
                split=conf["data"]["test_train_split"],
                percent_train=conf["data"]["percentage_train"],
            )

            print("Train values {}".format(tf.math.reduce_sum(train_data[1], axis=0)))
            print(
                "Validation values {}".format(
                    tf.math.reduce_sum(validation_data[1], axis=0)
                )
            )

            # Set up ML model
            input_shape = train_data[0].shape[-2:]
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

            save_copy_config(conf["save"]["config_dir"] + start_time, args.config_file)
            save_copy_config(
                conf["save"]["config_dir"] + start_time, conf["model"]["config_file"]
            )

            # Run training/learning algorithim
            history = fit_model(
                model, train_data, validation_data, callback_list, conf["fit"]
            )

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

            # Load test data
            data_files = get_file_list(
                conf["data"]["folder"] + "/" + conf["data"]["x_validation_exclude"]
            )
            test_data, test_label_data = load_data(
                data_files, conf["data"]["data_settings"]
            )

            test_data = np.asarray(test_data)
            test_label_data = np.asarray(test_label_data)

            # Analyse performance
            actual_labels = tf.argmax(test_label_data, axis=-1)
            predicted_classes = tf.argmax(model.predict(test_data), axis=-1)

            num_classes = conf["data"]["data_settings"]["num_labels"]

            TP = tf.math.count_nonzero(predicted_classes * actual_labels)
            TN = tf.math.count_nonzero((predicted_classes - 1) * (actual_labels - 1))
            FP = tf.math.count_nonzero(predicted_classes * (actual_labels - 1))
            FN = tf.math.count_nonzero((predicted_classes - 1) * actual_labels)

            accuracy = float(TP / actual_labels.shape[0])
            precision = float(TP / (TP + FP))
            recall = float(TP / (TP + FN))
            f1 = float(2 * precision * recall / (precision + recall))

            conf_matrix = tf.math.confusion_matrix(
                labels=actual_labels,
                predictions=predicted_classes,
                num_classes=num_classes,
            )

            print("Confusion matrix")
            print(conf_matrix)

            print("Accuracy")
            print(history.history["categorical_accuracy"][-1])
            print(history.history["val_categorical_accuracy"][-1])

            # Save results to log file
            with open(
                pathlib.Path(conf["hyper_paramaters"]["hparam_result_file"]), mode="a+"
            ) as file:
                print(
                    [
                        "Timestamp",
                        "epochs",
                        "train_accuracy",
                        "val_accuracy",
                        "test_accuracy",
                        "Precision",
                        "Recall",
                        "F1_Score",
                        "Conf_matrix",
                    ],
                    file=file,
                )
                print(
                    [
                        start_time,
                        history.epoch[-1],
                        history.history["categorical_accuracy"][-1],
                        history.history["val_categorical_accuracy"][-1],
                        accuracy,
                        precision,
                        recall,
                        f1,
                        np.array2string(
                            np.asarray(conf_matrix).reshape((-1)), separator=","
                        ),
                    ],
                    file=file,
                )
        # Catch any errors in code
        except KeyboardInterrupt:
            break
        # except Exception as e:
        #     print(e)
