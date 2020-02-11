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
BATCH_SIZE=128
CACHE=False
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
        if(i>0):
            break

    print("Number of all trainexamples: {}".format(len(trainExamples)))
    return trainExamples


def preprocess_data(split=0.9, cache=True):
    cacheFile = "/home/zombori/tmp/amoba_cache.npz"
    if cache and os.path.isfile(cacheFile):
        npz = np.load(cacheFile)
        train_xs = npz['train_xs']
        train_ys = npz['train_ys']
        test_xs = npz['test_xs']
        test_ys = npz['test_ys']
    else: 
        trainExamples = load_history()
        xs = []
        ys = []
        for (allBoard, curPlayer, pi, action) in trainExamples:
            # TODO understand action
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
        
        print("Input shape: ", xs.shape)
        print("Target shape: ", ys.shape)

        trainsize = int(len(trainExamples) * split)
        train_xs = xs[:trainsize]
        train_ys = ys[:trainsize]
        test_xs = xs[trainsize:]
        test_ys = ys[trainsize:]
        np.savez(cacheFile, train_xs=train_xs, train_ys=train_ys, test_xs=test_xs, test_ys=test_ys)

    print("Train input shape: ", train_xs.shape)
    print("Train output shape: ", train_ys.shape)
    print("Test input shape: ", test_xs.shape)
    print("Test output shape: ", test_ys.shape)

    return (train_xs, train_ys), (test_xs, test_ys)

(train_xs, train_ys), (test_xs, test_ys) = preprocess_data(cache=CACHE)

input_shape = train_xs.shape[1:]
output_shape = train_ys.shape[1:]
output_count = np.prod(output_shape)

# reaches around 25% accuracy
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=input_shape),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(output_count),
#     keras.layers.Reshape(output_shape)
# ])

inputs = keras.Input(shape=input_shape)
outputs = inputs
outputs = layers.Conv2D(NUM_CHANNELS, KERNEL_SIZE, activation='relu', padding="same")(outputs)
outputs = layers.Conv2D(NUM_CHANNELS, KERNEL_SIZE, activation='relu', padding="same")(outputs)
outputs = layers.Conv2D(NUM_CHANNELS, KERNEL_SIZE, activation='relu', padding="same", strides=(2,1))(outputs)
outputs = layers.Conv2D(NUM_CHANNELS, KERNEL_SIZE, activation='relu', padding="same", strides=(2,2))(outputs)
outputs = layers.Flatten()(outputs)

outputs_flat = layers.Flatten()(inputs)
outputs_flat = layers.Dense(512, activation='relu')(outputs_flat)
outputs_flat = layers.Dropout(DROPOUT_RATE)(outputs_flat)
outputs_flat = layers.Dense(256, activation='relu')(outputs_flat)
outputs_flat = layers.Dropout(DROPOUT_RATE)(outputs_flat)

outputs = layers.Concatenate(axis=1)([outputs_flat, outputs])
outputs = layers.Dense(1024, activation='relu')(outputs)
outputs = layers.Dropout(DROPOUT_RATE)(outputs)
outputs = layers.Dense(512, activation='relu')(outputs)
outputs = layers.Dropout(DROPOUT_RATE)(outputs)
pi = layers.Dense(output_count)(outputs)
prob = layers.Softmax()(pi)

model = keras.Model(inputs=inputs, outputs=pi)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)

# model = keras.Model(inputs=inputs, outputs=prob)
# loss = keras.losses.MeanSquaredError()

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['categorical_accuracy'])

model.fit(train_xs, train_ys, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_xs, test_ys))


# test_loss, test_acc = model.evaluate(test_xs,  test_ys, verbose=2)

print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)
