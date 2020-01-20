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
    print('Loading model...')
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph = graph)

    with tf.gfile.GFile("amoba_model.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    print('Check out the input placeholders:')
    nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
    for node in nodes:
        print(node)
    layers = [op for op in graph.get_operations()]
    for layer in layers:
        print(layer)

    
    sess.run("")

    
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
create_wrapper()
#create_and_save_model()
#graph()
