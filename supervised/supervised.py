import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os
from pickle import Pickler, Unpickler

print(tf.__version__)

EPOCHS = 200
NUM_CHANNELS=128
KERNEL_SIZE=(4,4)
DROPOUT_RATE=0
BATCHNORM=True
BATCH_SIZE=128
SPLIT=0.2
CACHE=True
REMOVE_DUPLICATES=False
AVERAGE_DUPLICATES=True
#DATAFILE="/home/doma945/amoba_teleport/temp_1000sim_medium"
DATAFILE="/home/doma945/amoba_teleport/temp_1000sim_big"
#DATAFILE="/home/doma945/amoba_teleport/temp_4000sim"
CACHEFILE = "/home/zombori/tmp/amoba_cache.npz"
NETWORK="linear"

def load_history():
    # ===== load history file =====
    # Descriptions: containes the data collected in every Iteration
    modelFile = os.path.join(DATAFILE, "trainhistory.pth.tar")
    examplesFile = modelFile+".examples"
    trainhistory = []
    if not os.path.isfile(examplesFile):
        print(examplesFile)
    else:
        print("File with trainExamples found. Read it.")
        with open(examplesFile, "rb") as f:
            for i in Unpickler(f).load():
                trainhistory.append(i)
        f.closed

    print("The trainhistory containes {} iteration of data".format(len(trainhistory)))

    # ===== Extract data =====
    trainExamples = []
    for i,e in enumerate(trainhistory):
        trainExamples.extend(np.array(e))

    print("Number of all trainexamples: {}".format(len(trainExamples)))
    return trainExamples

def remove_duplicates(xs, ps, vs):
    dict = {}
    for i in range(xs.shape[0]):
        s = str(xs[i])
        dict[s] = i

    indices = list(dict.values())
    xs2 = xs[indices]
    ps2 = ps[indices]
    vs2 = vs[indices]

    print("Reduced shapes {}, {}, {} to {}, {}, {}".format(xs.shape, ps.shape, vs.shape, xs2.shape, ps2.shape, vs2.shape))
    return xs2, ps2, vs2

def average_duplicates(xs, ps, vs):
    dict = {}
    for i in range(xs.shape[0]):
        s = str(xs[i])
        if s in dict:
            dict[s]["ps"].append(ps[i])
            dict[s]["vs"].append(vs[i])
        else:
            dict[s] = {"x": xs[i], "ps": [ps[i]], "vs": [vs[i]]}
    xs2 = []
    ps2 = []
    vs2 = []
    for s in dict:
        xs2.append(dict[s]["x"])
        ps2.append(np.mean(dict[s]["ps"], axis=0))
        vs2.append(np.mean(dict[s]["vs"]))
    xs2 = np.array(xs2)
    ps2 = np.array(ps2)
    vs2 = np.array(vs2)

    print("Average: reduced shapes {}, {}, {} to {}, {}, {}".format(xs.shape, ps.shape, vs.shape, xs2.shape, ps2.shape, vs2.shape))
    return xs2, ps2, vs2
    

def preprocess_data(cache=True):
    if cache and os.path.isfile(CACHEFILE):
        npz = np.load(CACHEFILE)
        xs = npz['xs']
        ps = npz['ps']
        vs = npz['vs']
    else: 
        trainExamples = load_history()
        xs = []
        ps = []
        vs = []
        curPlayers = []
        for (allBoard, curPlayer, pi, action) in trainExamples:
            xs.append(allBoard)
            curPlayers.append(curPlayer)
            ps.append(pi)
            vs.append(action)
        xs = np.array(xs)
        curPlayers = np.array(curPlayers)
        ps = np.array(ps)
        vs = np.array(vs)

        board = np.expand_dims(xs[:,:,:,0], axis = 3)
        heur_channels = xs[:,:,:,1:]
        white_board = board * (board+1) -1
        black_board = board * (board-1) -1

        curPlayers = curPlayers.reshape((-1, 1, 1, 1))
        player_channel = curPlayers * np.ones_like(board)            
        xs = np.concatenate([white_board, black_board, heur_channels, player_channel], axis=3)

        if AVERAGE_DUPLICATES:
            xs, ps, vs = average_duplicates(xs, ps, vs)
        elif REMOVE_DUPLICATES:
            xs, ps, vs = remove_duplicates(xs, ps, vs)
        np.savez(CACHEFILE, xs=xs, ps=ps, vs=vs)

    print("Input shape: ", xs.shape)
    print("Target policy shape: ", ps.shape)
    print("Target value shape: ", vs.shape)
    return (xs, ps, vs)


(xs, ps, vs) = preprocess_data(cache=CACHE)

def show(i):
    x = xs[i]
    p = ps[i]
    white = (x[:,:,0]+1) / 2
    black = (x[:,:,1]+1) / 2
    player = x[0,0,10]
    board = white - black
    policy = p.reshape((12,4))
    value = vs[i]
    print(np.transpose(board))
    print(np.transpose(policy))
    print("value: ", value)
    print("player: ", player)


# players = np.mean(xs[:,:,:,10], axis=(1,2))
# print(len(players))
# print(np.sum(players))


input_shape = xs.shape[1:]
policy_shape = ps.shape[1:]
pi_output_count = np.prod(policy_shape)

if NETWORK=="original":
    inputs = keras.Input(shape=input_shape)
    outputs = inputs
    outputs = layers.Conv2D(NUM_CHANNELS, KERNEL_SIZE, padding="same")(outputs)
    if BATCHNORM: outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = layers.Activation(tf.nn.relu)(outputs)
    outputs = layers.Conv2D(NUM_CHANNELS, KERNEL_SIZE, padding="same")(outputs)
    if BATCHNORM: outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = layers.Activation(tf.nn.relu)(outputs)
    outputs = layers.Conv2D(NUM_CHANNELS, KERNEL_SIZE, padding="same", strides=(2,1))(outputs)
    if BATCHNORM: outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = layers.Activation(tf.nn.relu)(outputs)
    outputs = layers.Conv2D(NUM_CHANNELS, KERNEL_SIZE, padding="same", strides=(2,2))(outputs)
    if BATCHNORM: outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = layers.Activation(tf.nn.relu)(outputs)
    outputs = layers.Flatten()(outputs)

    outputs_flat = layers.Flatten()(inputs)
    outputs_flat = layers.Dense(1512)(outputs_flat)
    if BATCHNORM: outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = layers.Activation(tf.nn.relu)(outputs)
    outputs_flat = layers.Dropout(DROPOUT_RATE)(outputs_flat)
    outputs_flat = layers.Dense(1256)(outputs_flat)
    if BATCHNORM: outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = layers.Activation(tf.nn.relu)(outputs)
    outputs_flat = layers.Dropout(DROPOUT_RATE)(outputs_flat)

    outputs = layers.Concatenate(axis=1)([outputs_flat, outputs])
    outputs = layers.Dense(1024)(outputs)
    outputs = layers.Dropout(DROPOUT_RATE)(outputs)
    if BATCHNORM: outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = layers.Activation(tf.nn.relu)(outputs)
    outputs = layers.Dense(512)(outputs)
    outputs = layers.Dropout(DROPOUT_RATE)(outputs)
    if BATCHNORM: outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = layers.Activation(tf.nn.relu)(outputs)
    pi = layers.Dense(pi_output_count, name="policy")(outputs)
    v0 = layers.Dense(1)(outputs)
    v = layers.Activation(tf.math.tanh, name="value")(v0)
    model = keras.Model(inputs=inputs, outputs=(pi, v))
elif NETWORK=="linear":
    inputs = keras.Input(shape=input_shape)
    pi = layers.Conv2D(1, (1,1), padding="same", name="conv1")(inputs)
    pi = layers.Flatten(name="policy")(pi)
    v = layers.Flatten()(inputs)
    # v = layers.Dense(1, activation="relu")(v)
    v = layers.Dense(1)(v)
    v = layers.Activation(tf.math.tanh, name="value")(v)
    model = keras.Model(inputs=inputs, outputs=(pi, v))
elif NETWORK=="linear2":
    inputs = keras.Input(shape=input_shape)
    pi = layers.Conv2D(10, (1,1), padding="same", name="conv1", activation="relu")(inputs)
    pi = layers.Conv2D(1, (1,1), padding="same", name="conv2")(pi)
    pi = layers.Flatten(name="policy")(pi)
    v = layers.Flatten()(inputs)
    # v = layers.Dense(1, activation="relu")(v)
    v = layers.Dense(1)(v)
    v = layers.Activation(tf.math.tanh, name="value")(v)
    model = keras.Model(inputs=inputs, outputs=(pi, v))
elif NETWORK=="local":
    inputs = keras.Input(shape=input_shape)
    outputs = layers.Conv2D(1, (3,3), padding="same", name="conv1")(inputs)
    pi = layers.Flatten(name="policy")(outputs)
    v0 = layers.Dense(1)(pi)
    v = layers.Activation(tf.math.tanh, name="value")(v0)
    model = keras.Model(inputs=inputs, outputs=(pi, v))
    
# elif NETWORK=="dense": # todo two heads
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=input_shape),
#         keras.layers.Dense(1128, activation='relu'),
#         keras.layers.Dense(1256, activation='relu'),
#         keras.layers.Dense(1128, activation='relu'),
#         keras.layers.Dense(output_count),
#         keras.layers.Reshape(output_shape)
#     ])


    
loss = {
    "policy": keras.losses.CategoricalCrossentropy(from_logits=True),
    "value": keras.losses.MeanSquaredError(),
}
loss_weights = {
    "policy": 1,
    "value": 10,
}
metrics = {
    "policy": ['categorical_accuracy',
               keras.metrics.TopKCategoricalAccuracy(2, "top2"),
               keras.metrics.TopKCategoricalAccuracy(3, "top3"),
               keras.metrics.TopKCategoricalAccuracy(4, "top4"),
               keras.metrics.TopKCategoricalAccuracy(5, "top5")],
    "value": 'mse',
}

# prob = layers.Softmax()(pi)
# model = keras.Model(inputs=inputs, outputs=prob)
# loss = keras.losses.MeanSquaredError()

model.compile(optimizer="adam",
              loss=loss,
              loss_weights=loss_weights,
              metrics=metrics
)

model.fit(xs, (ps, vs), epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=SPLIT, verbose=2)

mylayer = model.get_layer(name="conv1")
myweights = mylayer.trainable_weights[0]
myweights = myweights.numpy()[:,:,:,0]
print(myweights.shape)
for i in range(11):
    print("Filter ", i)
    print(myweights[:,:,i])
