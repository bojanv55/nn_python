import matplotlib.pyplot as plt
import imageio
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./datasets/MNIST_data/", one_hot=True)

model = keras.Sequential([
    keras.layers.Conv2D(32, (5,5), input_shape=(28, 28, 1), activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

input_layer = tf.reshape(mnist.train.images, [-1, 28, 28, 1])
lbls = [np.argmax(l) for l in mnist.train.labels]
lbls = np.reshape(lbls, 55000)

model.fit(input_layer, lbls, epochs=5, steps_per_epoch=1000)

input_t_layer = tf.reshape(mnist.test.images, [-1, 28, 28, 1])
lblst = [np.argmax(l) for l in mnist.test.labels]
lblst = np.reshape(lbls, 55000)

test_loss, test_acc = model.evaluate(input_t_layer, lblst)

print('Test accuracy:', test_acc)