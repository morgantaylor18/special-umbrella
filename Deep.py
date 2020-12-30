import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from tensorflow import keras


def unpickle(file):
    """Adapted from the CIFAR page: http://www.cs.utoronto.ca/~kriz/cifar.html"""
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


# Read the data
data_dir = '/home/users/drake/data/cifar-10-batches-py/' # BLT
data_dir = ''
train = [unpickle(data_dir + 'data_batch_{}'.format(i)) for i in [1, 2, 3, 4]]
X_train = np.stack([t[b'data'] for t in train])
X_train = tf.transpose(tf.reshape(X_train, [-1, 3, 32, 32]), (0, 2, 3, 1))
y_train = list(itertools.chain(*[t[b'labels'] for t in train]))
y_train = np.array(y_train)

batch5 = unpickle(data_dir + 'data_batch_5')
X_valid = batch5[b'data']
X_valid = tf.transpose(tf.reshape(X_valid, [-1, 3, 32, 32]), (0, 2, 3, 1))
y_valid = np.array(batch5[b'labels'])

labels = unpickle(data_dir + 'batches.meta')[b'label_names']

# Display one individual image and its label
# plt.imshow(X_train[0])
# plt.show()
# print(labels[y_train[0]])

# Build the network
# model = keras.models.Sequential([
#     keras.layers.conv(64, kernel_size=3, activation=tf.nn.relu, input_shape=(28,28,1)),
#     keras.layers.conv(32, kernel_size=3, activation=tf.nn.relu),
#     keras.layers.Flatten(),
#     keras.layers.Dense(10, activation='softmax')])

# model.add(tf.keras.layers.Dense(10,activation=tf.nn.relu))

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(5,5),activation=tf.nn.relu,
                                 input_shape=(32,32,3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer='adam',
              metrics=["accuracy"])

# Train the network
model.fit(X_train, y_train, epochs=40, validation_data=(X_valid, y_valid))

# Test the network on one image
predictions = model.predict_classes(X_train)


def check(i):
    plt.imshow(X_train[i])
    print(f'{predictions[i]}: {labels[predictions[i]]}')
    print(f'{y_train[i]}: {labels[y_train[i]]}')


check(42)


