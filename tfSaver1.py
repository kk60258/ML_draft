#!/usr/bin/env python
# -- coding: utf-8 --

import tensorflow as tf
import numpy as np
import argparse
import os
"""
define an easy linear regression model and use gradient descent to optimize to result.
train:
  train and save the trained weight(w) and bias(b) to .ckpt file
  
not train:
  restore weight and bias from ckpt file

"""

def train(isTrain=True):
    x = tf.placeholder(tf.float32, shape=[None, 1])
    y = 4 * x + 4

    w = tf.Variable(tf.random_normal([1], -1, 1))
    b = tf.Variable(tf.zeros([1]))
    y_predict = w * x + b

    loss = tf.reduce_mean(tf.square(y - y_predict))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    train_steps = 100
    checkpoint_steps = 50
    checkpoint_dir = os.getcwd() + '/tmp/tfsaver1/'
    directory = os.path.dirname(checkpoint_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
    x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))
    print('isTrain {}, checkpoint_dir {}'.format(isTrain, checkpoint_dir))
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if isTrain:
            summary_dir = os.getcwd() + '/tmp/tfsaver1_summary/'
            print('summary_dir {}'.format(summary_dir))
            summary_directory = os.path.dirname(summary_dir)
            if not os.path.exists(summary_directory):
                os.makedirs(summary_directory)

            writer = tf.summary.FileWriter(summary_directory, sess.graph)

            tf.summary.scalar("loss", loss)

            merged_summary = tf.summary.merge_all()

            for i in range(train_steps):
                _, summary = sess.run([train, merged_summary], feed_dict={x: x_data})

                if (i + 1) % checkpoint_steps == 0:
                    saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i + 1)

                writer.add_summary(summary, i)
            writer.flush()
            print('training completed')
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                pass
            print(sess.run(w))
            print(sess.run(b))

            print(sess.run(y_predict, feed_dict={x: x_data}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, help="0 means train; 1 means predict")
    args = parser.parse_args()
    if (args.train is None) :
        print("must provide --train 0 for training or --train 1 to see the trained result")
        exit()

    train(args.train == 0)
