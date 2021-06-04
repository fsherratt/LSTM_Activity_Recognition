"""
Pass original data through model and produce timeseries output
"""
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as K
import yaml
import os

from train import hardware_setup, Data_Preload, load_config

if __name__ == "__main__":
    SAVE_CLASS_PREDICTIONS = True
    SAVE_HIDDEN_STATE = False

    # z Gyro only
    # model_folder = "20201023-122111"  # W vs (SA / SD) - exclude [1/16/03/17] - Balanced dataset - 00.00%
    # model_folder = "20201023-131929"  # SA vs (W / SD) - exclude [1/16/03/17] - Balanced dataset - 81.95%
    # model_folder = "20201023-135657"  # W vs SA vs SD - exclude [1/16/03/17] - Balanced dataset - 57.71%

    # x Acc Only
    # model_folder = "20201026-183609"  # W vs (SA / SD) - 86.50%
    # model_folder = "20201027-101854"  # SA vs (W / SD) - 88.10%
    # model_folder = "20201027-102701"  # SD vs (W / SD) - 77.84%
    # model_folder = "20201027-103059"  # W vs SA vs SD - 72.62%

    # x Acc + y Gyro
    # model_folder = "20201102-153312"  # W vs (SA / SD) - 83.22%
    # model_folder = "20201102-155017"  # SD vs (W / SD) - 78.09%
    # model_folder = "20201102-182625"  # SA vs (W / SD) - 88.01%
    # model_folder = "20201102-183826"  # W vs SA vs SD - 65.38%

    # 6 axis IMU
    # model_folder = "20201103-103709"  # SA vs (W / SD) - 91.99%
    # model_folder = "20201103-142223"  # W vs (SA / SD) - 91.05%
    # model_folder = "20201103-143503"  # SD vs (W / SD) - 89.39%
    # model_folder = "20201103-145933"  # W vs SA vs SD - 59.38%
    model_folder = "20210325-112851"

    MODEL_FOLDER = pathlib.Path(model_folder)
    MODEL_DIR = pathlib.Path("logs\\model")
    CONF_DIR = pathlib.Path("logs\\conf")
    SAVE_DIR = pathlib.Path("result")

    # TEST_PARTICIPANT = ["Participant_04"]  # Part of training (Seen)
    # TEST_PARTICIPANT = ["Participant_07"]

    # TEST_PARTICIPANT = ["Participant_03"]  # Excluded participant (Unseen)
    TEST_PARTICIPANT = ["Participant_16"]

    conf = load_config(config_file=str(CONF_DIR / MODEL_FOLDER / "config.yaml"))
    conf["data"]["data_settings"]["skip"] = 0  # Don't skip anything

    # Load tensorflow model
    print("Loading model")
    model = tf.keras.models.load_model(str(MODEL_DIR / MODEL_FOLDER))
    print("Model loaded")
    model.summary()

    # Load Data
    data_files = Data_Preload(data_folder=conf["data"]["folder"], load_augment=False)

    if SAVE_CLASS_PREDICTIONS:
        prediction = []
        # mappings = {0: 0, 3: 0, 4: 0}
        # mappings = {0: 0, 1: 0, 2: 1}

        # for item in mappings.items():
        # conf["data"]["data_settings"]["label_mapping"] = {item[0]: item[1]}
        x, y = data_files.load_data(
            filter_mode=False,
            filter_list=TEST_PARTICIPANT,
            include_aug=False,
            parse_settings=conf["data"]["data_settings"],
        )
        prediction.append(model.predict(x, verbose=0))

        # Save to file
        if not os.path.exists(str(SAVE_DIR / MODEL_FOLDER)):
            os.mkdir(str(SAVE_DIR / MODEL_FOLDER))

        # Combine into a single matrix
        num_labels = conf["data"]["data_settings"]["num_labels"]
        rows = np.max([x.shape[0] for x in prediction])
        pred_array = np.full((rows, num_labels * len(prediction)), fill_value=pd.NA)

        for i in range(len(prediction)):
            pred_array[
                0 : prediction[i].shape[0],
                (num_labels * i) : ((num_labels * i) + num_labels),
            ] = prediction[i]

        pd.DataFrame(pred_array).to_csv(
            str(SAVE_DIR / MODEL_FOLDER / "class_pred.csv"), na_rep="NA"
        )
        pd.DataFrame(y).to_csv(str(SAVE_DIR / MODEL_FOLDER / "y.csv"), na_rep="NA")

    if SAVE_HIDDEN_STATE:
        conf["data"]["data_settings"]["label_mapping"] = None
        x, y = data_files.load_data(
            filter_mode=False,
            filter_list=TEST_PARTICIPANT,
            include_aug=False,
            parse_settings=conf["data"]["data_settings"],
        )

        # Copy weights to new model to extract hidden states
        tf_model = tf.keras.Sequential()
        tf_model.add(
            tf.keras.layers.LSTM(
                units=1,
                input_shape=model.layers[0].input_shape[-2:],
                return_sequences=True,
            )
        )
        tf_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
        tf_model.layers[0].set_weights(weights=model.layers[0].get_weights())

        hidden_state = tf_model.predict(x, verbose=0)

        hidden_state = np.squeeze(hidden_state)
        # hidden_state = np.reshape(hidden_state, [-1, 128])
        pd.DataFrame(hidden_state).to_csv(
            str(SAVE_DIR / MODEL_FOLDER / "hidden_state.csv")
        )
