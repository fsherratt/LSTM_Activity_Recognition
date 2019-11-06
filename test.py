import numpy as np

m = 0.2
c = 0.1

x = np.asarray(range(200), dtype=np.float32)
y = m*x + c

timesteps = 3

x_data = []
y_data = []
for i in range(100):
    x_data.append(x[i:i+3])
    y_data.append(y[i+5:i+8])

x_data = np.asarray(x_data)
y_data = np.asarray(y_data)

x_data = x_data[:,:,np.newaxis]
y_data = y_data[:,:,np.newaxis]

import tensorflow as tf

# define LSTM configuration
n_neurons = x_data.shape[1]
n_batch = x_data.shape[0]
n_epoch = 50000

# create LSTM
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(n_neurons, input_shape=x_data[0].shape, return_sequences=True))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))

model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())
# train LSTM
model.fit(x_data, y_data, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate

test_data = x[150:153]
test_data = np.reshape(test_data, (1,3,1))

result = model.predict(test_data, batch_size=1, verbose=0)
for value in result:
	print('%.1f' % value)

pass