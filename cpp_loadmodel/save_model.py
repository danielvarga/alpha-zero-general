import os
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import meta_graph

def create_and_save_model():
    inputs = tf.placeholder(tf.float32, shape=(1, 10), name='input')
    x = tf.keras.layers.Dense(5, activation='relu')(inputs)
    x = tf.keras.layers.Dense(5, activation='relu')(x)
    preds = tf.keras.layers.Dense(3, activation='tanh')(x)
    preds = tf.identity(preds, name="prediction")

    # Write the model definition
    with open('model.pb', 'wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())

def print_placeholders(filename, prefix = ""):
    f = open(filename, "r")
    graph_protobuf = text_format.Parse(f.read(), tf.GraphDef())
    shapes = {n.name:[d. size for d in n.attr['shape'].shape.dim] \
              for n in graph_protobuf.node if n.op in ('Placeholder')}
    print(shapes)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def=graph_protobuf, name="")

        sess = tf.Session(graph=graph)
        _ = tf.Variable(initial_value='fake_variable')
        init = tf.global_variables_initializer()
        
        sess.run(init)
        board = np.zeros((1,432))
        #board = np.zeros((1,12,4,9))
        curPlayer = np.ones((1,1))
        isTraining = False
        dropout = 0.0
        d1 = np.ones((1,49))
        d2 = np.ones((1,))

        v = graph.get_tensor_by_name(prefix +"v:0")
        prob = graph.get_tensor_by_name(prefix+"prob:0")
        board_tens = graph.get_tensor_by_name("input_boards:0")
        player_tens = graph.get_tensor_by_name(prefix+"curPlayer:0")
        dropout_tens = graph.get_tensor_by_name(prefix+"dropout:0")
        #training_tens = graph.get_tensor_by_name("isTraining:0")

        prob, v = sess.run([prob, v],
                        feed_dict={board_tens:board, player_tens:curPlayer,
                                   dropout_tens:dropout, #training_tens:isTraining,
                                   })
        print(prob, v)

        # === Save with ckeckpoint ===
        saver = tf.train.Saver()
        with open('model.pb', 'wb') as f:
            f.write(graph.as_graph_def().SerializeToString())
        saver.save(sess, "checkpoint/train.ckpt")


def save_and_load_frozen_graph():
    d1 = np.ones((1,10))
    d2 = np.ones((1,10))

    # === Save Simple Dense model: ===
    with tf.Graph().as_default() as graph:
        a = tf.placeholder(tf.float32, shape = (None,10), name = "a")
        b = tf.placeholder(tf.float32, shape = (None,10), name = "b")

        x = tf.keras.layers.Dense(1)(a)
        d= tf.identity(x, name = "out")
        c = tf.add(a,b, name = "out1")
        
        sess = tf.Session(graph=graph)
        init = tf.global_variables_initializer()
        sess.run(init) # === We need it because of the Dense layer ===
        print(sess.run([d], feed_dict={a:d1, b:d2}))

        # === Save with ckeckpoint ===
        saver = tf.train.Saver()
        with open('model.pb', 'wb') as f:
            f.write(graph.as_graph_def().SerializeToString())
        saver.save(sess, "checkpoint/train.ckpt")

        # === Save with frozen graph ===
        #  Var ==> Const
        input_graph_def = graph.as_graph_def()
        output_node_names = ['out', 'out1']
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_node_names)

        tf.io.write_graph(output_graph_def, './', 'example.pb')

    # === Load frozen Dense model ===
    f = open("example.pb", "r")
    graph_protobuf = text_format.Parse(f.read(), tf.GraphDef())
    with tf.Graph().as_default() as graph2:
        tf.import_graph_def(graph_def=graph_protobuf, name="")
        op = graph2.get_operations()
        print("Tensors: {}".format([m.values() for m in op]))

        inp1 =  graph2.get_tensor_by_name("a:0")
        inp2 =  graph2.get_tensor_by_name("b:0")
        out =  graph2.get_tensor_by_name("out:0")

        sess = tf.Session(graph=graph2)
        init = tf.global_variables_initializer()
        sess.run(init)
        res = sess.run([out], feed_dict={inp1:2*d1, inp2:2*d2})
        print("Result: {}".format(res))
        
def reshape_input(filename):
    print('Loading model...')
    f = open(filename, "r")
    graph_protobuf = text_format.Parse(f.read(), tf.GraphDef())
    shapes = {n.name:[d. size for d in n.attr['shape'].shape.dim] \
              for n in graph_protobuf.node if n.op in ('Placeholder')}
    inp_len = np.abs(np.prod(shapes['input_boards']))

    # === Reshape Input Model ===
    with tf.Graph().as_default() as graph1:
        input = tf.placeholder(tf.float32, (None, inp_len), name='input')
        x = tf.keras.layers.Reshape(shapes['input_boards'])(input)
        drop = tf.constant(0.0, dtype=tf.float32, shape=[], name='dropout')
        out = tf.identity(x, name='output')

    # === Big Model ===
    with tf.Graph().as_default() as graph2:
        tf.import_graph_def(graph_def=graph_protobuf, name="")

    # === Connect the 2 graphs: ===
    graph = tf.get_default_graph()
    x = tf.placeholder(tf.float32, (None, inp_len), name='input_boards')
    meta_graph1 = tf.train.export_meta_graph(graph=graph1)
    meta_graph.import_scoped_meta_graph(meta_graph1, input_map={'input': x}, import_scope='graph1')
    out1 = graph.get_tensor_by_name('graph1/output:0')
    out2 = graph.get_tensor_by_name('graph1/dropout:0')
    
    meta_graph2 = tf.train.export_meta_graph(graph=graph2)
    meta_graph.import_scoped_meta_graph(meta_graph2,
                                        input_map={'input_boards': out1,'dropout':out2},
                                        import_scope='graph2')

    # === Save with frozen ===
    frozen = False
    if frozen:
        input_graph_def = graph.as_graph_def()
        sess = tf.Session(graph = graph)
        output_node_names = ['graph2/v', 'graph2/prob']
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_node_names)

        tf.io.write_graph(output_graph_def, './', 'train_merged.pbtxt')
    else:
        tf.io.write_graph(graph, './', 'train_merged.pbtxt')


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
save_and_load_frozen_graph()
reshape_input("train.pbtxt")
#print_placeholders("train.pbtxt")
print_placeholders("train_merged.pbtxt", prefix = "graph2/")
