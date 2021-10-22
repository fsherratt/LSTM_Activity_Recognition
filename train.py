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


def load_model(model_dir):
    return tf.keras.models.load_model(model_dir)


def generate_model(input_shape):

    print("Input Shape: {}".format(input_shape))

    model = create_model(layer_definitions=model_conf["layers"], input_shape=input_shape)

    loss_func = loss_function(conf["loss_func"]["type"], conf["loss_func"]["settings"])
    model = compile_model(tf_model=model, loss_func=loss_func, settings=conf["compile"])

    return model


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
            epochs = (history.epoch[-1] + 1,)
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


if __name__ == "__main__":
    args = parse_cli()

    # Load configuration files
    _conf = load_config(config_file=args.config_file)
    _model_conf = load_config(config_file=_conf["model"]["config_file"])
    hparams = hparam_load.load_hparam_set(_conf["hyper_paramaters"]["hparam_file"])

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

        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Update config with next hyperparamater set
        conf = hparam_load.set_hparams(hparam_set, copy.deepcopy(_conf))
        model_conf = hparam_load.set_hparams(hparam_set, copy.deepcopy(_model_conf))

        # Print out hyperparamaters
        print("\n------------------------")
        print(f"Start: {start_time}")
        print(f"Sweep: {ix + 1} of {len(hparams)}")
        print(f"Hyperparamaters: {hparam_set}")
        print("------------------------\n")

        try:
            # Load data by sets of episodes
            # train_data, validation_data, test_data = load_data_episodes(conf, data_files)

            # Load data excluing participants
            train_d, validation_d, test_d = load_data_subjects(conf, data_files)
        except InsufficientData as exc:
            print(exc)
            continue

        # TODO: #1ht21uw automate the loading of pre-trainined models
        # Load an existing model
        # model_dir = pathlib.Path("logs/model/20211021-164227")  # 16 - unit model
        # model = load_model(model_dir)

        # Generate a new model
        input_shape = train_d[0].shape[-2:]
        model = generate_model(input_shape)

        # TODO: #1ht21uw pre-test model with training data and save result
        print("***Pre training testing***")
        # This is accuracy for test data
        actual_class, predicted_class = test_model(model, conf, train_d, validation_d, test_d)
        save_results(conf, hparam_set, None, True, actual_class, predicted_class, start_time)

        # Train model
        model, history = train_model(model, conf, train_d, validation_d, start_time)

        # Post training testing
        print("***Post training testing***")
        # This is accuracy for test data
        actual_class, predicted_class = test_model(model, conf, train_d, validation_d, test_d)
        save_results(conf, hparam_set, history, False, actual_class, predicted_class, start_time)
