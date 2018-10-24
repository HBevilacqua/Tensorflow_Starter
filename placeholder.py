import tensorflow as tf
import numpy

# Convert to an array
feeds = numpy.asarray([0,1,2,3,4])

# Constant variable "x"
x = tf.constant(1, dtype=tf.int16)

# placeholder as input (no need initializer, will be used as feed through "feed_dict")
i = tf.placeholder(tf.int16)

# a really (really) basic operation
model = tf.add(i,x)

# Perform the model operation in a context manager
with tf.Session() as sess:
	for feed in feeds:
		print(sess.run(model, feed_dict={i: feed}))