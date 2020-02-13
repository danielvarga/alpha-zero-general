import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os
from pickle import Pickler, Unpickler

print(tf.__version__)

EPOCHS = 10
NUM_CHANNELS=128
KERNEL_SIZE=(4,4)
DROPOUT_RATE=0
BATCHNORM=True
BATCH_SIZE=128
SPLIT=0.2
CACHE=False
REMOVE_DUPLICATES=False
DATAFILE="/home/doma945/amoba_teleport/temp"
# DATAFILE="amoba_samples/temp"

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

def remove_duplicates(x, y):
    dict = {}
    for i in range(x.shape[0]):
        s = str(x[i])
        dict[s] = i

    indices = list(dict.values())
    xs = x[indices]
    ys = y[indices]

    print("Reduced shapes {} and {} to {} and {}".format(x.shape, y.shape, xs.shape, ys.shape))
    return xs, ys


def preprocess_data(cache=True):
    cacheFile = "/home/zombori/tmp/amoba_cache.npz"
    if cache and os.path.isfile(cacheFile):
        npz = np.load(cacheFile)
        xs = npz['xs']
        ys = npz['ys']
    else: 
        trainExamples = load_history()
        xs = []
        ys = []
        for (allBoard, curPlayer, pi, action) in trainExamples:
            # TODO understand action is reward for value training
            xs.append(allBoard)
            ys.append(pi)
        xs = np.array(xs)
        ys = np.array(ys)

        board = np.expand_dims(xs[:,:,:,0], axis = 3)
        heur_channels = xs[:,:,:,1:]
        white_board = board * (board+1) -1
        black_board = board * (board-1) -1
        player_channel = curPlayer * np.ones_like(board)
        xs = np.concatenate([white_board, black_board, heur_channels, player_channel], axis=3)

        np.savez(cacheFile, xs=xs, ys=ys)

    print("Input shape: ", xs.shape)
    print("Target shape: ", ys.shape)
    if REMOVE_DUPLICATES:
        xs, ys = remove_duplicates(xs, ys)
    return (xs, ys)

(xs, ys) = preprocess_data(cache=CACHE)

input_shape = xs.shape[1:]
output_shape = ys.shape[1:]
output_count = np.prod(output_shape)

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=input_shape),
#     keras.layers.Dense(1128, activation='relu'),
#     keras.layers.Dense(1256, activation='relu'),
#     keras.layers.Dense(1128, activation='relu'),
#     keras.layers.Dense(output_count),
#     keras.layers.Reshape(output_shape)
# ])

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
pi = layers.Dense(output_count)(outputs)
prob = layers.Softmax()(pi)

model = keras.Model(inputs=inputs, outputs=pi)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)

# model = keras.Model(inputs=inputs, outputs=prob)
# loss = keras.losses.MeanSquaredError()

model.compile(optimizer="adam",
              loss=loss,
              metrics=['categorical_accuracy'])

model.fit(xs, ys, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=SPLIT)
