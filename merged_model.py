import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
import os
from pickle import Pickler, Unpickler


import Arena
from gobang.GobangGame import GobangGame, display
from gobang.GobangPlayers import *
from gobang.tensorflow.NNet import NNetWrapper as NNet


print(tf.__version__)

EPOCHS = 100
NUM_CHANNELS=128
KERNEL_SIZE=(4,4)
DROPOUT_RATE=0
BATCHNORM=True
BATCH_SIZE=128
SPLIT=0.2
CACHE=False
REMOVE_DUPLICATES=False
#DATAFILE="../temp_8_x4"
DATAFILE="/mnt/g1home/doma945/amoba_teleport/temp_1000sim_big"


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
    #print(trainExamples[0:10][0])

    arr = []
    for item in trainExamples[:]:
        arr.append(item[-3])

    print(np.mean(arr))
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
    cacheFile = "./amoba_cache.npz"
    if cache and os.path.isfile(cacheFile):
        npz = np.load(cacheFile)
        xs = npz['xs']
        ys = npz['ys']
    else: 
        trainExamples = load_history()
        xs = []
        ys = []
        vs = []
        for (allBoard, curPlayer, pi, action) in trainExamples:
            # TODO understand action is reward for value training
            xs.append(np.array(allBoard)[:,:,:])
            pi = np.array(pi)
            #max = np.max(pi)
            #pi[pi<max]=0
            #pi[pi==max]=1
            ys.append(pi[:])
            vs.append(action)
        xs = np.array(xs)
        ys = np.array(ys)
        vs = np.expand_dims(np.array(vs), axis = 1)

        board = np.expand_dims(xs[:,:,:,0], axis = 3)
        heur_channels = xs[:,:,:,1:]
        white_board = board * (board+1) -1
        black_board = board * (board-1) -1
        player_channel = curPlayer * np.ones_like(board)
        #xs = np.concatenate([white_board, black_board, heur_channels, player_channel], axis=3)
        xs = np.concatenate([heur_channels[:,:,:,1:]], axis=3)

        np.savez(cacheFile, xs=xs, ys=ys, vs=vs)

    print("Input shape: ", xs.shape)
    print("Target shape: ", ys.shape)
    print("Value shape: ", vs.shape)
    if REMOVE_DUPLICATES:
        xs, ys = remove_duplicates(xs, ys)
    return (xs, ys, vs)

def build_model0():
    inputs = keras.Input(shape=input_shape)
    outputs = inputs
    for i in range(3):
        outputs = layers.Conv2D(NUM_CHANNELS, KERNEL_SIZE, padding="same")(outputs)
        if BATCHNORM: outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = layers.Activation(tf.nn.relu)(outputs)

    outputs2 = layers.GlobalMaxPooling2D()(outputs)
    outputs2 = layers.Flatten()(outputs2) 
    v = layers.Dense(1, name = "v", activation = "tanh")(outputs2)
    #v = layers.Lambda(lambda x: tf.math.reduce_mean(x), name = "v")(outputs2)
    #v = tf.math.reduce_mean(outputs, name = "v")
    outputs = layers.Conv2D(64, (1,1), padding = "same")(outputs)
    outputs = layers.Activation('relu')(outputs)
    pi = layers.Conv2D(1, KERNEL_SIZE, padding="same")(outputs)
    pi = layers.Flatten(name = 'pi')(pi)
    prob = layers.Softmax()(pi)
    model = keras.Model(inputs=inputs, outputs=[pi, v, prob])

    return model


def build_model():
    inputs = keras.Input(shape=input_shape)
    outputs = layers.Flatten()(inputs)
    #x = layers.Dense(64, activation = "relu",
    #                 kernel_regularizer=regularizers.l2(0.01),
    #                 activity_regularizer=regularizers.l1(0.01) )(outputs)
    #x = layers.Dense(512, activation = "relu")(x)
    out = outputs
    v = layers.Dense(1, name = "v", activation = "tanh", 
                     kernel_regularizer=regularizers.l2(0.01),
                     activity_regularizer=regularizers.l1(0.01))(out)

    
    pi = layers.Dense(48, name="pi")(out)
    prob = layers.Softmax()(pi)
    model = keras.Model(inputs=inputs, outputs=[pi, v, prob])
    return model

class Model_Arena:
    def get_input_for_model(self, board):
        mtx = self.heuristic.get_field_stregth_mtx(board, 1)
        heuristic_components = self.heuristic.get_x_line_mtx(board, 1)
        shape = list(np.shape(board))+[1]
        new_board = np.concatenate([np.reshape(board,shape),
                                    np.reshape(mtx, shape),
                                    heuristic_components], axis=2)
        return heuristic_components

    def model_player(self,b,p, model):
        board = np.array([self.get_input_for_model(b)])
        valids = self.game.getValidMoves(b, 1)
        
        probs = model.predict(board)[2]
        move = np.argmax(probs*valids+valids*0.001)
        #print(probs, move)
        return move
        
    def __init__(self, model):
        self.game = GobangGame(col=12, row=4, nir=7, defender=-1)
        self.heuristic = Heuristic(self.game)
        heuristic_player = Heuristic(self.game).random_play
        model_player = lambda b, p: Model_Arena.model_player(self,b,p,model)
        self.arena = Arena.Arena(model_player, heuristic_player, self.game, display=display)

    def play(self, number_of_games=100):
        return self.arena.playGames(number_of_games, verbose=True)    

def play_arena(model):
    game = GobangGame(col=12, row=4, nir=7, defender=-1)
    heur = Heuristic(game)
    heuristic = Heuristic(game).random_play
    model_player = lambda b, p: np.argmax(model.predict(h.b))
    arena = Arena.Arena(model_player, heuristic,  game, display=display)
    return arena.playGames(100, verbose=True)



load_history()
exit()
# === MAIN ===
(xs, ys, vs) = preprocess_data(cache=CACHE)

#input_shape = xs.shape[1:]
input_shape = (12, 4, 7)

model = build_model()
loss = {#'pi':keras.losses.MeanSquaredError(),
        'pi':keras.losses.CategoricalCrossentropy(from_logits=True),
        'v':keras.losses.MeanSquaredError()}

print(model.summary())
model.compile(optimizer="adam",
              loss=loss,
              metrics={'pi':tf.keras.metrics.CategoricalAccuracy("acc")},
              loss_weights=[10, 1, 0])

model.fit(xs, [ys, vs], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=SPLIT)

arena = Model_Arena(model)
print(arena.play())
# === Big data ===
#DATAFILE="../temp"
#(xs_big, ys_big, vs_big) = preprocess_data(cache=CACHE)
#print(np.shape(xs_big))
#input_shape = (None, None, 11)
#model.fit(xs_big, [ys_big, vs_big], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=SPLIT)
