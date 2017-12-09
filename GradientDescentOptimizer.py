import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size = 100
epoch = 1000
display_epoch = 10

x_train = tf.placeholder(tf.float32, [None, 784])
y_train = tf.placeholder(tf.float32, [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x_train, w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y_train * tf.log(y), reduction_indices=[1]))
step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

y_test = tf.placeholder(tf.float32, [None, 10])

epoch_set = []
accuracy_set = []
with tf.Session() as session :
    session.run(tf.initialize_all_variables())
    for i in range(epoch):
        x_batch, y_batch = mnist_images.train.next_batch(batch_size)
        session.run(step, feed_dict={x_train:x_batch, y_train:y_batch})

        if i % display_epoch == 0:
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_test, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            epoch_set.append(i)
            accuracy_set.append(session.run(accuracy, feed_dict={x_train: mnist_images.test.images, y_test: mnist_images.test.labels}))

plt.plot(epoch_set, accuracy_set, 'o', label='MNIST simple optimizer')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_test,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(session.run(accuracy, feed_dict={x_train: mnist_images.test.images, y_test: mnist_images.test.labels}))