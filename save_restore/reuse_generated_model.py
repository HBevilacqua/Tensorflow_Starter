from __future__ import print_function
import tensorflow as tf

# -------- Getting the dataset --------

# Functions for downloading and reading MNIST data
from tensorflow.examples.tutorials.mnist import input_data
# Load the training and test data into constants
mnist = input_data.read_data_sets("/tmp/save_reuse_model/mnist", one_hot=True)

# Global variable initialization of the default graph
init = tf.global_variables_initializer()

# -------- Reusing the model --------

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Import graph and recover session
    saver = tf.train.import_meta_graph("/tmp/save_reuse_model/model.meta", clear_devices=True)
    saver.restore(sess, "/tmp/save_reuse_model/model")
    
    # Access the operations of activation from SAVED Collection
    accuracy = tf.get_collection('accuracy')[0]
    # othet options to access it, by using its name
    #activation = tf.get_default_graph.get_operation_by_name('the_name_of_your_activation_operations').outputs[0]
    
    prediction = tf.get_collection('prediction')[0]

    # Making a prediction by feeding placeholders as {'PlaceholderName:0': data}
    print("\nPrediction: ", sess.run(prediction, feed_dict={'X_feed:0': mnist.test.images,
                                                            'Y_feed:0': mnist.test.labels}))

# -------- Testing the trained model --------

    # Calculate accuracy for MNIST test images
    print("\nTesting Accuracy:", \
        sess.run(accuracy, feed_dict={  'X_feed:0': mnist.test.images,
                                        'Y_feed:0': mnist.test.labels}))
