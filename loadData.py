import pandas as pd
import numpy as np
import tensorflow as tf

def loadDataFile(filename, label_heading='Activity', timesteps=10, num_samples=None, num_activities=12):
    data = pd.read_csv(filename)
    label = data.pop(label_heading)

    label = tf.compat.v2.one_hot(label, num_activities)
    label = np.asarray(label)

    num_elements = data.shape[0]

    max_start_index = num_elements - timesteps
    
    if num_samples is None or num_samples > max_start_index:
        num_samples = max_start_index

    data_frames = []
    label_frames = []

    for i in range(num_samples):
        sample_end = i + timesteps
        data_frames.append(data.values[i:sample_end, :])
        label_frames.append(label[sample_end])

    return label_frames, data_frames

if __name__ == "__main__":
    BATCH_SIZE = None
    BUFFER_SIZE = 10000
    EVALUATION_INTERVAL = 200
    EPOCHS = 10

    TIMESTEPS = 25
    FEATURES = 6
    ACTIVITIES = 12

    test_data_path = 'UCI_HAD/Transform_Data/data_exp001.csv'
    y, x = loadDataFile(test_data_path, timesteps=TIMESTEPS, num_samples=BATCH_SIZE, num_activities=ACTIVITIES)
    BATCH_SIZE = len(y)

    x = np.asarray(x) # Samples, Timesteps, Feature
    y = np.asarray(y)

    train_test_split = int(0.7*BATCH_SIZE)

    np.random.seed(0)
    perms = np.random.permutation(x.shape[0])

    x_train, y_train = x.take(perms[0:train_test_split], axis=0), y.take(perms[0:train_test_split], axis=0)
    x_test,  y_test  = x.take(perms[train_test_split:],  axis=0), y.take(perms[train_test_split:],  axis=0)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(TIMESTEPS, return_sequences=True, input_shape=x_train[0].shape))
    model.add(tf.keras.layers.LSTM(TIMESTEPS, return_sequences=True))
    model.add(tf.keras.layers.LSTM(TIMESTEPS, return_sequences=True))
    model.add(tf.keras.layers.LSTM(TIMESTEPS, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(y_train[0].shape[0]))
    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    model.summary()

    model.fit(x_train, y_train, batch_size=x_train.shape[0], epochs=EPOCHS, validation_data=(x_test, y_test), validation_steps=x_test.shape[0])

    pass