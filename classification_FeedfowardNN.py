from __future__ import print_function
import tensorflow as tf
import numpy as np

# -------- Getting the dataset --------

# Functions for downloading and reading MNIST data
from tensorflow.examples.tutorials.mnist import input_data
# Load the training and test data into constants
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Displaying the first image
import matplotlib.pyplot as plt
first_image = np.array(mnist.train.images[0], dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.colorbar()
plt.show()

# -------- Defining the model IO --------

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph inputs
X_feed = tf.placeholder("float", [None, num_input])
Y_feed = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# -------- Creating the model --------

# Hidden fully connected layer with 256 neurons
layer_1 = tf.add(tf.matmul(X_feed, weights['h1']), biases['b1'])
# Hidden fully connected layer with 256 neurons
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
# Output fully connected layer with a neuron for each class
logits = tf.matmul(layer_2, weights['out']) + biases['out']

prediction = tf.nn.softmax(logits)

# -------- Defining the loss function --------

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y_feed))

# -------- Defining the optimization algorithm --------

train_optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# -------- Evaluating --------

# Vector of True/False 
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y_feed, 1))
# Mean (float32)correct_pred
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Global variable initialization of the default graph
init = tf.global_variables_initializer()

# -------- Training the model --------

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_optim, feed_dict={X_feed: batch_x, Y_feed: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss_train, acc_train = sess.run([loss, accuracy], feed_dict={	X_feed: batch_x,
                                                             				Y_feed: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss_train) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc_train))

    print("Optimization Finished!")

# -------- Testing the trained model --------

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={	X_feed: mnist.test.images,
        								Y_feed: mnist.test.labels}))

"""
- MNIST info:
-------------
http://yann.lecun.com/exdb/mnist/

- tf.reduce_mean:
-----------------
Computes the mean of elements across dimensions of a tensor.

- Loss: softmax_cross_entropy_with_logits (tf.nn):
--------------------------------------------------
Softmax: generalization of logistic regression used for multi-class classification
Cross entropy: Quantify the difference between two probability distributions
Tensorflow's logit is defined as the output of a neuron without applying activation function.

- AdamOptimizer class (tf.train):
----------------------------------
Optimizer that implements the Adam algorithm.
"minimize" method:
Adds operations to minimize loss by updating var_list.

- tf.equal:
-----------
Returns the truth value of (x == y) element-wise.

- tf.argmax:
------------
Returns the index with the largest value across axes of a tensor. 

- mnist.train.next_batch(batch_size):
-------------------------------------
Returns a tuple of two arrays, where the first represents a batch of 
batch_size MNIST images, and the second represents a batch of batch-size labels
corresponding to those images.
"""
