import tensorflow as tf
import numpy as np
"""
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

with tf.Session() as sess:
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('/tmp/mnist_convnet_model/model.ckpt-20000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    accuracy = sess.graph.get_collection_ref('accuracy')

    sess.run(accuracy, feed_dict={features : {"x": eval_data},label: eval_labels})

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, 'tagtag', '/tmp/mnist_convnet_model/')



mnist_classifier = tf.estimator.Estimator(model_dir="/tmp/mnist_convnet_model")

eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
"""