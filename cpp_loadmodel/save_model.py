import tensorflow as tf
import os

def graph():
    graph = tf.Graph()
    with graph.as_default():
        input = tf.placeholder(tf.float32, [10,1], name='input')
        output = tf.reduce_sum(input**2, name='output')
        tf.train.write_graph(graph, '.', 'model2.pb')
    print("Saved graph")
    
def create_and_save_model():
    inputs = tf.placeholder(tf.float32, shape=(1, 10), name='input')
    x = tf.keras.layers.Dense(5, activation='relu')(inputs)
    x = tf.keras.layers.Dense(5, activation='relu')(x)
    preds = tf.keras.layers.Dense(3, activation='tanh')(x)
    preds = tf.identity(preds, name="prediction")

    #saver = tf.train.Saver()
    #with open('model3.pb', 'wb') as f:
    #    f.write(tf.get_default_graph().as_graph_def().SerializeToString())

    #init_op = tf.global_variables_initializer()
    #with tf.Session() as sess:
    #    sess.run(init_op)
    #    saver.save(sess, "checkpoint/train.ckpt")

    i = tf.initializers.global_variables()
    # Write the model definition
    with open('model.pb', 'wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())

def create_wrapper():
    from google.protobuf import text_format
    import numpy as np
    
    print('Loading model...')
    f = open("train.pbtxt", "r")
    graph_protobuf = text_format.Parse(f.read(), tf.GraphDef())

    graph_clone = tf.Graph()
    with graph_clone.as_default():
        tf.import_graph_def(graph_def=graph_protobuf, name="")

    #print(graph_clone.as_graph_def())

    print('Check out the input placeholders:')
    nodes = [n.name + ' => ' +  n.op for n in graph_protobuf.node if n.op in ('Placeholder')]
    shapes = [[d. size for d in n.attr['shape'].shape.dim] for n in graph_protobuf.node if n.op in ('Placeholder')]
    for node in nodes:
        print(node)
    print(shapes)
    print(tf.global_variables())
    layers = [op for op in graph_clone.get_operations()]
    #for layer in layers:
    #    print(layer)

def reshape_input(filename):
    from google.protobuf import text_format
    
    print('Loading model...')
    f = open(filename, "r")
    graph_protobuf = text_format.Parse(f.read(), tf.GraphDef())

    with tf.Graph().as_default() as graph1:
        input = tf.placeholder(tf.float32, (None, 9), name='flat_board')
        output = tf.identity(input, name='output')

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
#reshape_input("train.pbtxt")
create_wrapper()
#create_and_save_model()
#graph()
