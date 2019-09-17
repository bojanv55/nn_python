import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./datasets/MNIST_data/", one_hot=True)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(5,5), input_shape=(28, 28, 1), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

input_layer = tf.reshape(mnist.train.images, [-1, 28, 28, 1])
input_layer = np.reshape(mnist.train.images, (55000, 28, 28, 1))
#lbls = tf.reshape(mnist.train.labels, [-1, 10])

y_ = tf.reshape(mnist.train.labels, [-1, 10])
y_ = mnist.train.labels

x_test = tf.reshape(mnist.test.images, [-1, 28, 28, 1])
y_test = tf.reshape(mnist.test.labels, [-1, 10])

model.fit(input_layer, y_,
          batch_size=128,
          epochs=10,
          verbose=1,
          steps_per_epoch=20,
          validation_steps=1,
          validation_data=(x_test, y_test))

# input_t_layer = tf.reshape(mnist.test.images, [-1, 28, 28, 1])
# lblst = [np.argmax(l) for l in mnist.test.labels]
# lblst = np.reshape(lblst, 55000)
#
# test_loss, test_acc = model.evaluate(input_t_layer, lblst)
#
# print('Test accuracy:', test_acc)

score = model.evaluate(x_test, y_test, verbose=1, steps=20)
print('Test loss:', score[0])
print('Test accuracy:', score[1])