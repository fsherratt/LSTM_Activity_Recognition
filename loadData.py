import pandas as pd
import numpy as np
import tensorflow as tf
import os
import datetime

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
    # Setup Tensorflow enviroment
    USE_GPU = False
    # Disable GPU
    if not USE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



    # Load training data
    TIMESTEPS = 128
    FEATURES = 6
    ACTIVITIES = 12

    test_data_path = 'H:\\dos\\PhD\\tensorflow\\LSTM_Activity_Recognition\\UCI_HAD\\Transform_Data\\data_exp001.csv'
    y, x = loadDataFile(test_data_path, timesteps=TIMESTEPS, num_activities=ACTIVITIES)

    # Normalise data

    samples_available = len(y) 

    x = np.asarray(x) # Samples, Timesteps, Feature
    y = np.asarray(y)

    train_test_split = int(0.7*samples_available)

    np.random.seed(0)
    perms = np.random.permutation(x.shape[0])

    x_train, y_train = x.take(perms[0:train_test_split], axis=0), y.take(perms[0:train_test_split], axis=0)
    x_test,  y_test  = x.take(perms[train_test_split:],  axis=0), y.take(perms[train_test_split:],  axis=0)



    # Build Model
    ENABLE_DROPOUT = True
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(TIMESTEPS, return_sequences=True, input_shape=x_train[0].shape))
    model.add(tf.keras.layers.LSTM(TIMESTEPS, return_sequences=True))
    model.add(tf.keras.layers.LSTM(TIMESTEPS, return_sequences=True))
    model.add(tf.keras.layers.LSTM(TIMESTEPS, return_sequences=False))
    if ENABLE_DROPOUT:
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(y_train[0].shape[0]))
    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.summary()



    # Create callbacks
    # Save model weights
    PERIOD = 5
    checkpoint_path = "training\\cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    weight_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, period=PERIOD)

    model.save_weights(checkpoint_path.format(epoch=0)) 

    # Save tensorboard logs
    log_dir = "logs\\scalars\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Early stopping
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=20)

    cb_list = [weight_callback, tensorboard_callback, earlystop_callback]



    # Train Model
    EPOCHS = 1000
    BATCH_SIZE = 1000
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, 
              validation_data=(x_test, y_test), validation_steps=1,
              callbacks=cb_list)



    # Save Model
    model.save_weights(checkpoint_path.format(epoch=EPOCHS)) 