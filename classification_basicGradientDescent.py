from __future__ import print_function
import tensorflow as tf
import numpy as np
from csv import reader

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def create_layer_params(shape):
	#initial = tf.constant(0.5, shape=shape)
	#return tf.Variable(initial)
	return tf.Variable(tf.random_normal(shape))




# Load the data from csv file
# One data set: [i1, i2, i3, i4, i5, i6, i7, class]
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)

# Convert string numbers to floats, eg: '15.26' to 15.26
for idx_column in range(len(dataset[0])-1):
	str_column_to_float(dataset, idx_column)

# Convert class columns to integers, eg: '1' to 0, '2' to 2 and '3' to 1
# [12.3, 13.34, 0.8684, 5.243, 2.974, 5.637, 5.063, '3']
# to
# [12.3, 13.34, 0.8684, 5.243, 2.974, 5.637, 5.063, 1]
str_column_to_int(dataset, len(dataset[0])-1)

# Find the min and max values for each column, minmax =
# [[10.59, 21.18], [12.41, 17.25], [0.8081, 0.9183], [4.899, 6.675], [2.63, 4.033], [0.7651, 8.456], [4.519, 6.55], [0, 2]]
minmax = dataset_minmax(dataset)

# Normalize input variables to the range of 0 and 1 (range of the transfer function)
normalize_dataset(dataset, minmax)

# Parameters
l_rate = 0.5
n_epoch = 500
n_inputs_dataset = 7
n_classes = 3
n_rows_dataset = np.array(dataset).shape[0]
n_neurons_hidden = 5
n_test = 50
n_train = n_rows_dataset - n_test

# -------- Defining the dataset --------

train_x = np.zeros(shape=(n_train, n_inputs_dataset))
train_y = np.zeros(shape=(n_train, n_classes))
test_x = np.zeros(shape=(n_test, n_inputs_dataset))
test_y = np.zeros(shape=(n_test, n_classes))

# Create train_x ans train_y
for i_train in range(n_train):
	for i_input in range(n_inputs_dataset):
		train_x[i_train][i_input] = dataset[i_train][i_input]
	# One hot encoding
	train_y[i_train][dataset[i_train][-1]] = 1

# Create test_x ans test_y
for i_test in range(n_test):
	for i_input in range(n_inputs_dataset):
		test_x[i_test][i_input] = dataset[i_test][i_input]
	# One hot encoding
	test_y[i_test][dataset[i_test][-1]] = 1

# Create weights
# 7 inputs * x neurons 
w_hidden 	= create_layer_params([n_inputs_dataset, n_neurons_hidden])
# x neurons * 3 outputs 
w_out 		= create_layer_params([n_neurons_hidden, n_classes])

# Create biases
# one / hidden neuron
b_hidden 	= create_layer_params([n_neurons_hidden])
# one / classe
b_out 		= create_layer_params([n_classes])

# tf Graph inputs
X_feed = tf.placeholder(tf.float32, shape=[None, n_inputs_dataset], name="X_feed")
Y_feed = tf.placeholder(tf.float32, shape=[None, n_classes], name="Y_feed")

# -------- Creating the model --------

hidden_layer = tf.layers.dense(X_feed, units=n_inputs_dataset, activation=tf.nn.relu)
prediction = tf.layers.dense(hidden_layer, units=n_classes, activation=tf.nn.softmax)

# -------- Defining the loss function --------

# Loss function: Mean Squared Error (MSE)
loss_op = tf.reduce_sum(tf.pow(prediction-Y_feed, 2))/(2*n_train)

# -------- Defining the optimization algorithm --------

optim = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
# Add operations to the graph to minimize a cost by updating a list of variables.
optim_op = optim.minimize(tf.cast(loss_op, "float"))

# -------- Evaluating --------

# Vector of True/False: ARGMAX(prediction) == ARGMAX(Y_feed)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y_feed, 1))
# Mean (float32)correct_pred
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Global variable initialization of the default graph
init = tf.global_variables_initializer()

# -------- Training the model --------

with tf.Session() as sess:
	# Run the initializer
	sess.run(init)

	print("Inital hidden weights = ", sess.run(w_hidden))
	print("Inital out weights = ", sess.run(w_out))
	print("\nTrain:")
	for epoch in range(n_epoch):
		sess.run(optim_op, feed_dict={X_feed: train_x, Y_feed: train_y})
		if epoch % 50 == 0 or epoch == 0:
			loss_train, acc_train = sess.run([loss_op, accuracy_op], feed_dict={X_feed: train_x, Y_feed: train_y})
			print("Loss= {:.4f}".format(loss_train) + ", Accuracy= {:.4f}".format(acc_train))

# -------- Testing the trained model --------

	loss_test_op = tf.reduce_sum(tf.pow(prediction-Y_feed, 2))/(2*n_test)
	loss_test, acc_test = sess.run([loss_test_op, accuracy_op], feed_dict={X_feed: test_x, Y_feed: test_y})
	print("\nTest:")
	print("Loss = {:.4f}".format(loss_test))
	print("Accuracy = {:.4f}".format(acc_test))