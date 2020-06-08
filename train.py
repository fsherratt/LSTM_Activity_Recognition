"""
This file setups and run ML training based of provided config files
"""
import argparse
import datetime
import os
import pathlib

import numpy as np
import omegaconf
import pandas as pd
import tensorflow as tf
import yaml


def parse_cli():
    parser = argparse.ArgumentParser(description="Startup module(s)")
    parser.add_argument(
        "-c", "-C", "--config-file", type=str, default=None, help="Config file path",
    )
    return parser.parse_known_args()[0]


def load_config(config_file) -> omegaconf.dictconfig.DictConfig:
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
    }

    # Add input definition to first layer
    layer_definitions[0]["args"]["input_shape"] = input_shape

    tf_model = tf.keras.Sequential()

    for layer in layer_definitions:
        tf_model.add(layer_dict[layer["type"]](**layer["args"]))

    return tf_model


def compile_model(settings: dict, tf_model: tf.keras.Sequential) -> tf.keras.Sequential:
    tf_model.compile(**settings)
    tf_model.summary()

    return tf_model


def fit_model(
    tf_model: tf.keras.Sequential,
    training_data,
    validation_data,
    callbacks: None,
    settings=None,
):
    # TODO: parameterise constants
    defaults = {"batch_size": 2000, "epochs": 1000, "validation_steps": 1}
    if settings is None:
        settings = defaults
    else:
        settings = {**defaults, **settings}

    tf_model.fit(
        x=training_data[0],
        y=training_data[1],
        validation_data=validation_data,
        callbacks=callbacks,
        **settings
    )


def get_file_list(data_directory: str) -> list:
    data_directory = pathlib.Path(data_directory)
    files = os.listdir(data_directory)
    data_files = []

    # Use every other file - leave one blind sample for each user
    for file in files:
        file = data_directory / file
        if file.exists() and file.suffix == ".csv":
            data_files.append(file)

    return data_files


def load_data(data_files: list):
    x = []
    y = []

    for file in data_files:
        print("Opening {}".format(file))
        # TODO: parameterise constants
        new_x, new_y = parse_file(
            file, label_heading="activity", num_timesteps=128, num_labels=12, skip=5
        )
        x.extend(new_x)
        y.extend(new_y)

    x = np.asarray(x)
    y = np.asarray(y)

    return x, y


def parse_file(filename, label_heading, num_timesteps, num_labels, skip):
    # Read csv file
    data = pd.read_csv(filename)

    # Apply one_hot to labels
    label = data.pop(label_heading)
    label = tf.compat.v2.one_hot(label, num_labels)
    label = np.asarray(label)

    # Divide data up into timestep chunks - offset by skip steps
    data_frames = []
    label_frames = []

    i = 0
    max_start_index = data.shape[0] - num_timesteps

    while i < max_start_index:
        sample_end = i + num_timesteps

        data_frames.append(data.values[i:sample_end, :])
        label_frames.append(label[sample_end])

        i += skip

    return data_frames, label_frames


def split_test_train(data, labels, split=0.7):
    samples_available = len(labels)
    train_test_split = int(split * samples_available)

    np.random.seed(0)
    perms = np.random.permutation(data.shape[0])

    train = (
        data.take(perms[0:train_test_split], axis=0),
        labels.take(perms[0:train_test_split], axis=0),
    )
    test = (
        data.take(perms[train_test_split:], axis=0),
        labels.take(perms[train_test_split:], axis=0),
    )

    return test, train


def tensorboard_callback():
    # TODO: parameterise constants
    log_dir = "logs\\scalars\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)


def early_stopping_callback():
    # TODO: parameterise constants
    settings = {"monitor": "val_loss", "verbose": 1, "patience": 3}
    return tf.keras.callbacks.EarlyStopping(**settings)


def save_model_callback():
    # TODO: parameterise constants
    settings = {
        "filepath": "training\\cp-{epoch:04d}.ckpt",
        "verbose": 1,
        "save_weights_only": False,
        "period": 1,
    }
    return tf.keras.callbacks.ModelCheckpoint(**settings)


if __name__ == "__main__":
    args = parse_cli()

    # Setup physical devices
    try:
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e)

    # Load configuration files
    conf = load_config(config_file=args.config_file)
    model_conf = load_config(config_file=conf["model"]["config_file"])

    # Import and process data
    data_files = get_file_list(conf["data"]["folder"])
    raw_data, label_data = load_data(data_files)

    test_data, train_data = split_test_train(raw_data, label_data, split=0.7)

    # Set up ML model
    data_shape = train_data[0].shape[-2:]
    model = create_model(layer_definitions=model_conf["layers"], input_shape=data_shape)
    model = compile_model(settings=conf["compile"], tf_model=model)

    # Setup callbacks
    callback_list = []

    callback_list.append(tensorboard_callback())
    callback_list.append(early_stopping_callback())
    callback_list.append(save_model_callback())

    # Run training/learning algorithim
    fit_model(model, train_data, test_data, callback_list)

    # Save model and model properties
    foldername = "test"
    model.save(foldername)
