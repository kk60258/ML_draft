import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)
total_size = mnist_images.train.num_examples
batch_size = 100
epoch = 1
display_epoch = 2
learning_rate = 0.005

n_input = 784
n_hidder_1 = 100
n_hidder_2 = 100
n_classes = 10


x_train = tf.placeholder(tf.float32, [None, n_input])
y_train = tf.placeholder(tf.float32, [None, n_classes])

#hidden layer 1
#w_1 = tf.Variable(tf.zeros([n_input, n_hidder_1]))
w_1 = tf.Variable(tf.random_normal([n_input, n_hidder_1]))
#b_1 = tf.Variable(tf.zeros([n_hidder_1]))
b_1 = tf.Variable(tf.random_normal([n_hidder_1]))
z_1 = tf.nn.relu(tf.add(tf.matmul(x_train, w_1), b_1))
#z_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_train, w_1), b_1))


#hidden layer 2
#w_2 = tf.Variable(tf.zeros([n_hidder_1, n_hidder_2]))
w_2 = tf.Variable(tf.random_normal([n_hidder_1, n_hidder_2]))
#b_2 = tf.Variable(tf.zeros([n_hidder_2]))
b_2 = tf.Variable(tf.random_normal([n_hidder_2]))
z_2 = tf.nn.relu(tf.add(tf.matmul(z_1, w_2), b_2))
#z_2 = tf.nn.sigmoid(tf.add(tf.matmul(z_1, w_2), b_2))

#output layer
#w_out = tf.Variable(tf.zeros([n_hidder_2, n_classes]))
w_out = tf.Variable(tf.random_normal([n_hidder_2, n_classes]))
#b_out = tf.Variable(tf.zeros([n_classes]))
b_out = tf.Variable(tf.random_normal([n_classes]))
z_out = tf.nn.softmax(tf.add(tf.matmul(z_2, w_out), b_out))

#cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=z_out, labels=y_train))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z_out, labels=y_train))
step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

epoch_set = []
accuracy_set = []
y_test = tf.placeholder(tf.float32, [None, n_classes])
num_batch = int(total_size / batch_size)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for num_epoch in range(epoch):
        for _ in range(num_batch):
            x_batch, y_batch = mnist_images.train.next_batch(batch_size)
            sess.run(step, feed_dict={x_train: x_batch, y_train: y_batch})

        correct_prediction = tf.equal(tf.argmax(z_out, 1), tf.argmax(y_test, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        epoch_set.append(num_epoch)
        accuracy_set.append(
            sess.run(accuracy, feed_dict={x_train: mnist_images.test.images, y_test: mnist_images.test.labels}))

    # Save the variables to disk.
    saver = tf.train.Saver
    save_path = saver.save(sess=sess, save_path="/home/nineg/model.ckpt")
    print("Model saved in file: %s" % save_path)

plt.plot(epoch_set, accuracy_set, 'o', label='MNIST two hidden layer optimizer')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

