import tensorflow as tf

## ----- Creating variables -----

variable = tf.get_variable("variable", [1, 2, 3])
variable_int = tf.get_variable("variable_int", [1, 2, 3], dtype=tf.int32)
variable_initilalized = tf.get_variable("variable_init", [1, 2, 3], initializer=tf.zeros_initializer)

#print(variable) ### <tf.Variable 'variable:0' shape=(1, 2, 3) dtype=float32_ref>

sess = tf.Session()

sess.run(tf.global_variables_initializer())

#print(variable) ### <tf.Variable 'variable:0' shape=(1, 2, 3) dtype=float32_ref>

print(sess.run(variable)) ### [[[-0.95292616 -0.8183949   0.19033766]  [-0.33327454  0.8850558  -0.13136292]]]
print(sess.run(variable_int)) ### [[[0 0 0]  [0 0 0]]]
print(sess.run(variable_initilalized)) ### [[[0. 0. 0.]  [0. 0. 0.]]]



## ----- Variable collection -----

# trainable (default) - variables that can be shared across multiple devices
var_trainable = tf.get_variable("var_trainable", shape=(), collections=[tf.GraphKeys.GLOBAL_VARIABLES])

# non trainable - variables for which TensorFlow will calculate gradients.
var_nonTrainable1 = tf.get_variable("var_nonTrainable1", shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])
var_nonTrainable2 = tf.get_variable("var_nonTrainable", shape=(), trainable=False)

# create collection
tf.add_to_collection("my_collection", var_nonTrainable1)
# local initializer
sess.run(tf.local_variables_initializer())

print(tf.get_collection("my_collection")) ### [<tf.Variable 'var_nonTrainable1:0' shape=() dtype=float32_ref>]
print(sess.run(tf.get_collection("my_collection"))) ### [-0.367555]



## ----- Using variables -----

var = tf.get_variable("var", shape=(), initializer=tf.zeros_initializer())
# assignment: 1.0
assignment = var.assign_add(1)
sess.run(tf.global_variables_initializer())

print(assignment) ### Tensor("AssignAdd:0", shape=(), dtype=float32_ref)
# sess.run(assignment) or assignment.op.run(), or assignment.eval()
print(sess.run(assignment)) ### [1.0]

var_2 = var.read_value()
print(sess.run(var_2)) ### [1.0]



## ----- Sharing variables -----

# function which creates two variables as it is called
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

# variable puts into "conv1" for the first concolutional relu layer, into "conv2" for the second layer
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])

input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])

"""
# Code wich failes:
x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
print("before failing")
# This fails: 	Since the desired behavior is unclear (create new variables 
#				or reuse the existing ones?) TensorFlow will fail

x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])
"""

# to reuse variables
with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)

# or
"""
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables()
  output2 = my_image_filter(input2)
"""


## ----- Sources -----

# https://www.tensorflow.org/guide/variables
