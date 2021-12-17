"""
This file setups and run ML training based of provided config files
"""
import argparse
import copy
import datetime
import os
import pathlib


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import Sigmoid
import yaml

import hparam_load
from data_files import Data_Files, InsufficientData
from model import *


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
    return parser.parse_args()


def load_config(config_file) -> dict:
    with open(config_file) as file:
        config = yaml.full_load(file)

    return config


def save_copy_config(file_dir: str, config_file: str, config: dict):
    config_file = pathlib.Path(config_file)
    file_dir = pathlib.Path(file_dir)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    file_path = file_dir / config_file.name
    with open(file_path, "w") as file:
        file.write(yaml.dump(config))


def save_list_to_file(file, values):
    """
    @param file: str
        File to save to

    @param header: list
        List of header values to save to file
    """
    with open(pathlib.Path(file), mode="a") as file:
        file.write(",".join(map(str, values)) + "\n")


def load_data_episodes(conf: dict, data_files: Data_Files):
    """
    @raises: InsufficientData
    """
    # Import and process data
    # Filter out participant for cross validation study
    train_samples = int(conf["data"]["training_samples"] * conf["data"]["test_train_split"])
    valid_samples = int(conf["data"]["training_samples"] * (1 - conf["data"]["test_train_split"]))
    data = data_files.load_data_activities(
        parse_kwargs=conf["data"]["data_settings"],
        file_offset=conf["data"]["episode_offset"],
        train_samples=train_samples,
        valid_samples=valid_samples,
        test_samples=conf["data"]["test_samples"],
        filter_mode=False,  # Only Include participants in list
        filter_list=conf["data"]["x_validation_exclude"],
    )

    return prepare_data(conf, data)


def load_data_subjects(conf: dict, data_files: Data_Files):
    # TODO: #1ht27r1 Add in leave one out cross validation
    data = data_files.load_data(
        parse_kwargs=conf["data"]["data_settings"],
        validation_split=conf["data"]["test_train_split"],
        filter_mode=True,  # Exclude participants in list
        filter_list=conf["data"]["x_validation_exclude"],
    )

    return prepare_data(conf, data)


def prepare_data(conf: dict, data: dict):

    train_data = data["train"]
    validation_data = data["valid"]
    test_data = data["test"]

    # Convert to one hot
    if train_data is not None:
        print("Training Data Samples: {}".format(len(train_data[1])))

        train_data_labels = label_to_one_hot(
            train_data[1],
            conf["data"]["data_settings"]["num_labels"],
            conf["data"]["data_settings"]["label_mapping"],
        )
        train_data = (train_data[0], train_data_labels)

    if validation_data is not None:
        print("Validation Data Samples: {}".format(len(validation_data[1])))

        validation_data_labels = label_to_one_hot(
            validation_data[1],
            conf["data"]["data_settings"]["num_labels"],
            conf["data"]["data_settings"]["label_mapping"],
        )
        validation_data = (validation_data[0], validation_data_labels)

    if test_data is not None:
        print("Test Data Samples: {}".format(len(test_data[1])))

        test_data_labels = label_to_one_hot(
            test_data[1],
            conf["data"]["data_settings"]["num_labels"],
            conf["data"]["data_settings"]["label_mapping"],
        )
        test_data = (test_data[0], test_data_labels)

    return train_data, validation_data, test_data


def generate_model(conf: dict, model_conf: dict, input_shape):

    print("Input Shape: {}".format(input_shape))

    model = create_model(layer_definitions=model_conf["layers"], input_shape=input_shape)

    loss_func = loss_function(conf["loss_func"]["type"], conf["loss_func"]["settings"])
    model = compile_model(tf_model=model, loss_func=loss_func, settings=conf["compile"])

    return model


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def train_model(model, conf, train_data, validation_data, start_time):
    # ------------------------------------------------------------------------------
    # Setup callbacks
    callback_list = []

    if conf["callbacks"]["use_tensorboard"]:
        tensorboard_dir = pathlib.Path(conf["save"]["tensorboard_dir"] + "/" + start_time)
        callback_list.append(
            tensorboard_callback(
                settings=conf["callbacks"]["tensorboard"], save_dir=tensorboard_dir,
            )
        )

    if conf["callbacks"]["use_early_stopping_threshold"]:
        callback_list.append(
            early_stopping_threshold_callback(
                stop_threshold=conf["callbacks"]["early_stopping_threshold"]
            )
        )

    if conf["callbacks"]["use_early_stopping"]:
        callback_list.append(early_stopping_callback(settings=conf["callbacks"]["early_stopping"]))

    if conf["callbacks"]["use_save_model"]:
        model_save_dir = pathlib.Path(conf["save"]["model_dir"] + "/" + start_time + "/")
        callback_list.append(
            save_model_callback(settings=conf["callbacks"]["save_model"], save_path=model_save_dir,)
        )

    if conf["callbacks"]["use_learning_rate_scheduler"]:
        callback_list.append(tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1))

    # ------------------------------------------------------------------------------
    # Print list of classes
    print("Using callbacks: {}".format(", ".join(x.__class__.__name__ for x in callback_list)))

    if conf["save"]["config"]:
        save_folder = conf["save"]["config_dir"] + start_time

        save_copy_config(save_folder, "config.yaml", conf)
        save_copy_config(
            save_folder, "model.yaml", model_conf,
        )

    # ------------------------------------------------------------------------------
    # Run training/learning algorithim
    history = fit_model(model, train_data, validation_data, callback_list, conf["fit"])

    # Save final model
    # ------------------------------------------------------------------------------
    model_save_dir = pathlib.Path(conf["save"]["model_dir"] + "/" + start_time + "/")
    if conf["save"]["final_model"]:
        # Save final model and model properties
        model.save(model_save_dir.__str__())

        # Save model history
        history_save_dir = pathlib.Path(conf["save"]["history_dir"] + "/" + start_time + ".csv")
        pd.DataFrame.from_dict(history.history).to_csv(history_save_dir.__str__(), index=False)

    return model, history


def test_model(model, conf, train_data, validation_data, test_data):
    """
    @param test_data: np.array
        None if not test data
    """
    # Print data summary
    print("*** Train Performance ***")
    gen_conf_matrix(
        model, train_data[0], train_data[1], conf["data"]["data_settings"]["num_labels"],
    )
    print("*** Validation Performance ***")
    actual_class, predicted_class = gen_conf_matrix(
        model, validation_data[0], validation_data[1], conf["data"]["data_settings"]["num_labels"],
    )

    if test_data is not None:
        print("*** Test Performance ***")
        actual_class, predicted_class = gen_conf_matrix(
            model, test_data[0], test_data[1], conf["data"]["data_settings"]["num_labels"],
        )

    return actual_class, predicted_class


def save_results(conf, hparam_set, history, pre_test, actual_class, predicted_class, start_time):
    """
    @param pre_test: bool
        True to indicate testing was performed before training. False otherwise
    """
    # TODO: #1ht3uvj - save confusion matricies

    if conf["hyper_paramaters"]["save_hparam"]:
        # Analyse performance
        class_report = analyse_test_data(
            y_true=actual_class,
            y_pred=predicted_class,
            num_classes=conf["data"]["data_settings"]["num_labels"],
        )

        # Save results to log files
        # Make participant exclusion data CSV friendly
        if hparam_set["HP_VALIDATION_EXCLUDE"] is not None:
            hparam_set["HP_VALIDATION_EXCLUDE"] = "-".join(
                map(str, hparam_set["HP_VALIDATION_EXCLUDE"])
            )

        header = [
            "Timestamp",
            "Sweep",
            "Sweep Total",
            "Pre Test",
            *hparam_set.keys(),
        ]
        values = [start_time, ix + 1, len(hparams), pre_test, *hparam_set.values()]
        save_list_to_file(conf["hyper_paramaters"]["hparam_log_file"], header)
        save_list_to_file(conf["hyper_paramaters"]["hparam_log_file"], values)

        # Results
        header = [
            "Timestamp",
            "params",
            "epochs",
            "train_accuracy",
            "val_accuracy",
            *class_report.keys(),
        ]

        if history is None:
            epochs = 0
            cat_acc = 0
            val_acc = 0
        else:
            epochs = history.epoch[-1] + 1
            cat_acc = history.history["categorical_accuracy"][-1]
            val_acc = history.history["val_categorical_accuracy"][-1]

        values = [
            start_time,
            model.count_params(),
            epochs,
            cat_acc,
            val_acc,
            *class_report.values(),
        ]
        save_list_to_file(conf["hyper_paramaters"]["hparam_result_file"], header)
        save_list_to_file(conf["hyper_paramaters"]["hparam_result_file"], values)


_model_cache = {}


def load_model(model_dir):
    """
    Cache ML models
    """
    if model_dir not in _model_cache:
        print("Loading model from disk")
        model = tf.keras.models.load_model(model_dir)
        _model_cache[model_dir] = model

    print("Model cached... Cloning")
    return tf.keras.models.clone_model(_model_cache[model_dir])


if __name__ == "__main__":
    args = parse_cli()

    # Load configuration files
    _conf = load_config(config_file=args.config_file)
    _model_conf = load_config(config_file=_conf["model"]["config_file"])
    hparams = hparam_load.load_hparam_set(_conf["hyper_paramaters"]["hparam_file"])

    # Setup physical devices
    hardware_setup(**_conf["hardware_setup"])

    data_files_general = Data_Files(
        data_folder=_conf["data"]["folder_gen"],
        verbose=_conf["data"]["verbose"],
        shuffle=True,
        preload=False,
    )

    data_files = Data_Files(
        data_folder=_conf["data"]["folder"],
        verbose=_conf["data"]["verbose"],
        shuffle=True,
        preload=False,
    )

    # Hyper paramater training
    stop_ix = None
    start_ix_offset = 84

    for ix, hparam_set in enumerate(hparams):
        if ix + 1 < start_ix_offset:
            continue

        if stop_ix is not None and ix >= stop_ix:
            exit(1)

        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Update config with next hperparamater set
        conf = hparam_load.set_hparams(hparam_set, copy.deepcopy(_conf))
        model_conf = hparam_load.set_hparams(hparam_set, copy.deepcopy(_model_conf))

        # Print out hyperparamaters
        print("\n------------------------")
        print(f"Start: {start_time}")
        print(f"Sweep: {ix + 1} of {len(hparams)}")
        print(f"Hyperparamaters: {hparam_set}")
        print("------------------------\n")

        # Ignore models that only have 1 LSTM layer - delete this as it would still allow the dense network to be retrained
        # if not model_conf["layers"][0]["enabled"] and not model_conf["layers"][2]["enabled"]:
        #     print("Model does not contain layers for transfer learning. Skipping")
        #     continue

        try:
            # Load data by sets of episodes
            train_data_target, valid_data_target, test_data_target = load_data_episodes(
                conf, data_files
            )

            # Load data excluing participants
            # train_data_gen, validation_data_gen, _ = load_data_subjects(conf, data_files_general)

            # # Combine data sets
            # train_data = (
            #     np.concatenate((train_data_gen[0], train_data_target[0])),
            #     np.concatenate((train_data_gen[1], train_data_target[1])),
            # )
            # validation_data = (
            #     np.concatenate((validation_data_gen[0], valid_data_target[0])),
            #     np.concatenate((validation_data_gen[1], valid_data_target[1])),
            # )
            train_data = train_data_target
            validation_data = valid_data_target
            test_data = test_data_target

        except InsufficientData as exc:
            print(exc)
            continue
        # Generate a new model
        input_shape = train_data[0].shape[-2:]
        model = generate_model(conf, model_conf, input_shape)

        # TRANSFER LEARNING
        # Load an existing model
        model_dir = (
            pathlib.Path(conf["base_model"]["folder"]) / conf["base_model"]["model_name"]
        )  # 32 - unit model
        inner_model = load_model(model_dir)
        # Only retrain LSTM layer

        # Wrap pre-trained model with a new Dense input layer
        # model = tf.keras.Sequential()
        # model.add(tf.keras.Input(shape=input_shape))
        # model.add(tf.keras.layers.LSTM(units=6, return_sequences=True))
        # model.add(inner_model)
        # model.build(input_shape=(None, input_shape[0], input_shape[1]))
        # model.summary()

        # loss_func = loss_function(conf["loss_func"]["type"], conf["loss_func"]["settings"])
        # model = compile_model(tf_model=model, loss_func=loss_func, settings=conf["compile"])

        # transfer_layer = 0  # if model_conf["layers"][0]["enabled"] else 0
        # print(f"Transfer layer {transfer_layer}")

        # Copy across dense weights
        model.layers[3].set_weights(weights=inner_model.layers[3].get_weights())
        model.layers[3].trainable = False
        model.summary()

        # Set the starting learning rate
        tf.keras.backend.set_value(model.optimizer.learning_rate, conf["learning_rate"])

        # This is accuracy for test data
        test_pre_train = False
        if test_pre_train:
            print("***Pre training testing***")
            actual_class, predicted_class = test_model(
                model, conf, train_data_target, valid_data_target, test_data_target
            )
            save_results(conf, hparam_set, None, True, actual_class, predicted_class, start_time)

        # Train model
        model, history = train_model(model, conf, train_data, validation_data, start_time)

        # Post training testing
        print("***Post training testing***")
        # This is accuracy for test data
        actual_class, predicted_class = test_model(
            model, conf, train_data, validation_data, test_data
        )
        save_results(conf, hparam_set, history, False, actual_class, predicted_class, start_time)
        
        print("Cleaning up")
        tf.keras.backend.clear_session()

