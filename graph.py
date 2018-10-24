import tensorflow as tf

## ----- Creating graph 1 -----

graph1 = tf.Graph()
# "graph1" as context manager object
with graph1.as_default():
	# Operations created in this scope will be added to "graph1"
	c = tf.constant("Node in graph1")


## ----- Creating graph 2 -----

graph2 = tf.Graph()
with graph2.as_default():
	# Operations created in this scope will be added to "graph2"
	d1 = tf.constant(1)
	d2 = tf.constant(2)
	dRes = tf.add(d1,d2)

	# print: [<tf.Operation 'Const' type=Const>, <tf.Operation 'Const_1' type=Const>, <tf.Operation 'Add' type=Add>]
	print(graph2.get_operations())

	# Sessions created in this scope will run operations from graph2
  	sess2 = tf.Session()


## ----- Retrieving graph 1 / session 1 -----

# Sessions will run operations from "graph1"
sess1 = tf.Session(graph=graph1)

e1 = tf.constant(3, name='e1')
e2 = tf.constant(4, name='e2')
eRes = tf.add(e1,e2, name='add_e1e2')


## ----- Getting default graph -----

getgraph = tf.get_default_graph()

# Print all of the operations in the default graph
# print: [<tf.Operation 'e1' type=Const>, <tf.Operation 'e2' type=Const>, <tf.Operation 'add_e1e2' type=Add>]
print(getgraph.get_operations())
