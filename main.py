from Coach import Coach
from gobang.GobangGame import GobangGame as Game
from gobang.tensorflow.NNet import NNetWrapper as nn
from utils import *
import os

"""
Before using multiprocessing, please check 2 things before use this script.
1. The number of PlayPool should not over your CPU's core number.
2. Make sure all Neural Network which each process created can store in VRAM at same time. Check your NN size before use this.
"""
args = dotdict({
    'multiCPU': True,
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 30,
    'updateThreshold': 0.51,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 100,
    'cpuct': 4.0,
    'multiGPU': True,
    'setGPU': '3,4',
    # The total number of games when self-playing is:
    # Total = numSelfPlayProcess * numPerProcessSelfPlay
    'numSelfPlayProcess': 8,
    'numPerProcessSelfPlay': 20,
    # The total number of games when against-playing is:
    # Total = numAgainstPlayProcess * numPerProcessAgainst
    'numAgainstPlayProcess': 8,
    'numPerProcessAgainst': 5,
    'checkpoint': './temp/',
    'numItersForTrainExamplesHistory': 20,
})

if __name__=="__main__":
    import tensorflow as tf
    # tf.logging.info('TensorFlow')
    tf.logging.set_verbosity(tf.logging.ERROR)
    # tf.logging.info('TensorFlow')
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    g = Game(col=8, row=4, nir=7, defender=-1)
    c = Coach(g, args)
    c.learn()
