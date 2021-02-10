import os

import numpy as np
import pandas as pd
import tensorflow as tf

from train import Data_Preload, load_config, label_to_one_hot

NO_TRAN_FOLDER = (
    "C:/Users/Freddie/Documents/PhD/Data/Results/"
    "18_20201116_1129_transition_state/Data/100hz_no_tran"
)

LOG_FOLDER = (
    "C:/Users/Freddie/Documents/PhD/Machine-Learning/LSTM_Activity_Recognition/logs/"
)

SAVE_DIR = (
    "C:/Users/Freddie/Documents/PhD/Machine-Learning/LSTM_Activity_Recognition/result/"
)

# Load normal and transition data
# tran_data_files = Data_Preload(
#     data_folder=TRAN_FOLDER, load_augment=False, verbose=False,
# )
no_tran_data_files = Data_Preload(
    data_folder=NO_TRAN_FOLDER, load_augment=False, verbose=False,
)

# Available models
models_files = Data_Preload.get_file_list(LOG_FOLDER + "model", ".pb")

for mdl_file in models_files:
    mdl_name = mdl_file.parent.name
    # -------------------------------------------------------------------------------------------- #
    # Load data set
    print("*** Loading Config ***")
    conf = load_config(LOG_FOLDER + "conf/" + mdl_name + "/config.yaml")

    exclude = conf["data"]["x_validation_exclude"]
    settings = conf["data"]["data_settings"]
    settings["skip"] = 10

    # if settings["num_labels"] == 6:
    #     TRANSITION_MODEL = False
    #     print("Ingoring non-transition models")
    #     continue

    # -------------------------------------------------------------------------------------------- #
    # Load model
    print("*** Loading model {} ***".format(mdl_name))
    model = tf.keras.models.load_model(str(mdl_file.parent))

    # -------------------------------------------------------------------------------------------- #
    # Load data set
    print("*** Loading Data ***")
    # Validation Data
    train_data, train_label = no_tran_data_files.load_data(
        parse_settings=settings,
        filter_mode=True,
        filter_list=exclude,
        include_aug=False,
    )
    train_label = label_to_one_hot(
        train_label, settings["num_labels"], settings["label_mapping"],
    )

    # Test Data
    test_data, test_label = no_tran_data_files.load_data(
        parse_settings=settings,
        filter_mode=False,
        filter_list=exclude,
        include_aug=False,
    )
    test_label = label_to_one_hot(
        test_label, settings["num_labels"], settings["label_mapping"],
    )

    # -------------------------------------------------------------------------------------------- #
    # Pass both through data for test/validation sets
    print("*** Generating Confusion Matrices ***")

    print("*** Train Predictions ***")
    actual_class = tf.math.argmax(train_label, axis=-1)
    train_pred = model.predict(train_data)
    predicted_class = tf.math.argmax(train_pred, axis=-1)

    print("*** Training Confusion Matrix ***")
    train_conf_matrix = tf.math.confusion_matrix(
        labels=actual_class,
        predictions=predicted_class,
        num_classes=conf["data"]["data_settings"]["num_labels"],
    )
    print(train_conf_matrix)

    print("*** Test Predictions ***")
    actual_class = tf.math.argmax(test_label, axis=-1)
    test_pred = model.predict(test_data)
    predicted_class = tf.math.argmax(test_pred, axis=-1)

    print("*** Test Confusion Matrix ***")
    test_conf_matrix = tf.math.confusion_matrix(
        labels=actual_class,
        predictions=predicted_class,
        num_classes=conf["data"]["data_settings"]["num_labels"],
    )
    print(test_conf_matrix)

    # -------------------------------------------------------------------------------------------- #
    # Save confusion matrix
    print("*** Saving results ***")
    SAVE_PATH = SAVE_DIR + mdl_name

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    pd.DataFrame(np.asarray(train_label)).to_csv(str(SAVE_PATH + "/train_act.csv"))
    pd.DataFrame(np.asarray(train_pred)).to_csv(str(SAVE_PATH + "/train_pred.csv"))
    pd.DataFrame(np.asarray(test_label)).to_csv(str(SAVE_PATH + "/test_act.csv"))
    pd.DataFrame(np.asarray(test_pred)).to_csv(str(SAVE_PATH + "/test_pred.csv"))

    pd.DataFrame(np.asarray(train_conf_matrix)).to_csv(
        str(SAVE_PATH + "/train_conf_mat.csv")
    )
    pd.DataFrame(np.asarray(test_conf_matrix)).to_csv(
        str(SAVE_PATH + "/test_conf_mat.csv")
    )
# EOF
