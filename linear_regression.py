from __future__ import print_function
import tensorflow as tf
import numpy as np

# -------- Defining the dataset --------

# Inputs
Xtrain = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# Expected outputs
Ytrain = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
2.827,3.465,1.65,2.904,2.42,2.94,1.3])
# Test
Xtest = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
Ytest = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

# 1 Row
nbTrainSamples = Xtrain.shape[0]

# -------- Defining the model --------

# Training parameters
learning_rate = 0.01
training_epochs = 1000

# Graph inputs
X_feed = tf.placeholder("float")
Y_feed = tf.placeholder("float")

# Sets model weights and bias (rank 0 tensor)
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Linear model = XW + b
Ypred = tf.add(tf.multiply(X_feed, W), b)

# -------- Defining the loss function --------

# Loss function: Mean Squared Error (MSE)
loss = tf.reduce_sum(tf.pow(Ypred-Y_feed, 2))/(2*nbTrainSamples)

# -------- Defining the optimization algorithm --------

# Gradient descent
# Note: minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Global variable initialization of the default graph
init = tf.global_variables_initializer()

# -------- Training the model --------

# Session context manager
with tf.Session() as sess:
    # Run the initializer
	sess.run(init)

	for epoch in range(training_epochs):
		# Trains the linear model with the training data
		for (xTrain, yTrain) in zip(Xtrain, Ytrain):
			sess.run(optimizer, feed_dict={X_feed: xTrain, Y_feed: yTrain})
		# Displays the loss during the training
		if (epoch+1) % 100 == 0:
			c = sess.run(loss, feed_dict={X_feed: Xtrain, Y_feed: Ytrain})
			print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(c), \
            "W=", sess.run(W), "b=", sess.run(b))

	training_loss = sess.run(loss, feed_dict={X_feed: Xtrain, Y_feed: Ytrain})
	print("Training loss=", training_loss, "W=", sess.run(W), "b=", sess.run(b))

# -------- Testing the trained model --------
	lost_test = tf.reduce_sum(tf.pow(Ypred-Y_feed, 2)) / (2 * Xtest.shape[0])
	testing_loss = sess.run(lost_test, feed_dict={X_feed: Xtest, Y_feed: Ytest})
	print("Testing loss=", testing_loss)

"""
- np.random.rand:
-----------------
Creates an array of the given shape and populates it with random 
samples from a uniform distribution over.

- tf.reduce_sum (method):
-------------------------
Computes the sum of elements across dimensions of a tensor. 
(deprecated arguments)

- zip:
------
for (xTrain, yTrain) in zip(Xtrain, Ytrain):
Iterate over two lists in parallel.

Code:
alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']

for a, b in zip(alist, blist):
    print a, b

Results:
a1 b1
a2 b2
a3 b3

- tf.train (module):
--------------------
Support for training models.

- GradientDescentOptimizer class (tf.train):
--------------------------------------------
Optimizer that implements the gradient descent algorithm.
"minimize" method:
Adds operations to minimize loss by updating var_list.
"""