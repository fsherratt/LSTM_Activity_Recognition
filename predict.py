"""
Pass original data through model and produce timeseries output
"""
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from train import get_file_list, load_data, parse_file, hardware_setup


if __name__ == "__main__":
    hardware_setup(use_gpu=True, random_seed=0)

    # Load data file
    data_dir = "C:/Users/Freddie/Documents/PhD/Machine-Learning/Output/9_20200723_110700_Stop_State_No_Tran_No_Normalize/Predict_Data/"
    data_dir = pathlib.Path(data_dir)

    file_list = get_file_list(data_dir)
    if not file_list:
        Exception("No files found")

    # Todo load this from config file
    settings = {
        "num_timesteps": 128,
        "label_heading": "activity",
        "normalize": False,
        "data_headings": [
            "r_ankle_accel_x",
            "r_ankle_accel_y",
            "r_ankle_accel_z",
            "r_ankle_gyro_x",
            "r_ankle_gyro_y",
            "r_ankle_gyro_z",
            "l_ankle_accel_x",
            "l_ankle_accel_y",
            "l_ankle_accel_z",
            "l_ankle_gyro_x",
            "l_ankle_gyro_y",
            "l_ankle_gyro_z",
        ],
        "num_labels": 4,
        "skip": 0,
    }

    data, label = load_data(file_list, settings, append=True)

    data_pairs = zip(file_list, data)

    # Load tensorflow model
    print("Loading model")
    model_dir = "C:/Users/Freddie/Documents/PhD/Machine-Learning/Output/9_20200723_110700_Stop_State_No_Tran_No_Normalize/Model/model/"
    model_dir = pathlib.Path(model_dir).__str__()

    model = tf.keras.models.load_model(model_dir)
    print("Model loaded")
    model.summary()

    # Pass timeseries data through model and store result
    for f, d in data_pairs:
        d = np.asarray(d)
        result = model.predict(d)

        # Save results to CSV file
        save_dir = "result/"
        save_file = pathlib.Path(save_dir + "Predict_" + f.stem + ".csv").__str__()
        print("Saving: {}".format(save_file))
        np.savetxt(save_file, result, delimiter=",")
