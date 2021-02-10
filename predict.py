"""
Pass original data through model and produce timeseries output
"""
import pathlib

import numpy as np
import pandas as pd

import tensorflow as tf

from train import Data_Preload, load_config, label_to_one_hot, split_test_train


if __name__ == "__main__":
    LOG_DIR = pathlib.Path("logs")
    MODEL_DIR = LOG_DIR / "model"
    CONF_DIR = LOG_DIR / "conf"
    DATA_DIR = pathlib.Path(
        (
            "C:/Users/Freddie/Documents/PhD/Data/Results/"
            "18_20201116_1129_transition_state/Data/100hz_no_tran"
            # "18_20201116_1129_transition_state/Data/100hz_transition"  # Transition state will be excluded!
        ),
    )
    SAVE_DIR = pathlib.Path("result")

    # Preload data
    data_files = Data_Preload(data_folder=DATA_DIR, load_augment=False, verbose=False,)

    model_files = Data_Preload.get_file_list(str(MODEL_DIR), ".pb")

    for mdl_file in model_files:
        np.random.seed(1)
        mdl_name = mdl_file.parent.name

        print("*** Model {} ***".format(mdl_name))

        print("*** Loading Config ***")
        conf = load_config(str(CONF_DIR / mdl_name / "config.yaml"))

        exclude = conf["data"]["x_validation_exclude"]
        settings = conf["data"]["data_settings"]
        settings["skip"] = 10

        print(settings["data_headings"])
        print(settings["label_mapping"])
        # Load model
        print("*** Loading model ***")
        model = tf.keras.models.load_model(str(mdl_file.parent))

        train_data, train_label = data_files.load_data(
            parse_settings=settings,
            filter_mode=True,
            filter_list=exclude,
            include_aug=False,
        )
        # _, train_data = split_test_train(
        #     train_data, train_label, split=1.0, percent_train=1.0, max_difference=None,
        # )
        # train_label = train_data[1]
        # train_data = train_data[0]
        train_label = label_to_one_hot(
            train_label, settings["num_labels"], settings["label_mapping"],
        )

        # Test Data
        test_data, test_label = data_files.load_data(
            parse_settings=settings,
            filter_mode=False,
            filter_list=exclude,
            include_aug=False,
        )
        # _, test_data = split_test_train(
        #     test_data, test_label, split=1.0, percent_train=1.0, max_difference=None,
        # )
        # test_label = test_data[1]
        # test_data = test_data[0]
        test_label = label_to_one_hot(
            test_label, settings["num_labels"], settings["label_mapping"],
        )

        # Run data through model
        print("*** Generating Confusion Matrices ***")

        print("*** Train Predictions ***")
        actual_class = tf.math.argmax(train_label, axis=-1)
        train_pred = model.predict(train_data)
        predicted_class = tf.math.argmax(train_pred, axis=-1)
        acc = (
            sum(np.asarray(predicted_class) == np.asarray(actual_class))
            / predicted_class.shape[0]
        )

        print("*** Training Confusion Matrix ***")
        train_conf_matrix = tf.math.confusion_matrix(
            labels=actual_class,
            predictions=predicted_class,
            num_classes=conf["data"]["data_settings"]["num_labels"],
        )
        print(train_conf_matrix)
        print("Accuracy: {}".format(acc))
        pd.DataFrame(np.asarray(train_conf_matrix)).to_csv(
            str(SAVE_DIR / (mdl_name + "_nt_train_conf.csv"))
        )

        print("*** Test Predictions ***")
        actual_class = tf.math.argmax(test_label, axis=-1)
        test_pred = model.predict(test_data)
        predicted_class = tf.math.argmax(test_pred, axis=-1)
        acc = (
            sum(np.asarray(predicted_class) == np.asarray(actual_class))
            / predicted_class.shape[0]
        )

        print("*** Test Confusion Matrix ***")
        test_conf_matrix = tf.math.confusion_matrix(
            labels=actual_class,
            predictions=predicted_class,
            num_classes=conf["data"]["data_settings"]["num_labels"],
        )
        print(test_conf_matrix)
        print("Accuracy: {}".format(acc))
        pd.DataFrame(np.asarray(test_conf_matrix)).to_csv(
            str(SAVE_DIR / (mdl_name + "_nt_test_conf.csv"))
        )

        print("*-------------------------------------------------------------*")
