from Coach import Coach
from gobang.GobangGame import GobangGame as Game
from utils import *
import os

"""
Before using multiprocessing, please check 2 things before use this script.
1. The number of PlayPool should not over your CPU's core number.
2. Make sure all Neural Network which each process created can store in VRAM at same time. Check your NN size before use this.
"""
args = dotdict({
    'displaybar': True,
    'multiCPU': True,
    'numIters': 10,
    'numEps': 100,
    'tempThreshold': 48,
    'updateThreshold': 0.51,
    'maxlenOfQueue': 320000,
    'numMCTSSims': 600,
    'cpuct': 3.5,
    'multiGPU': False,
    'setGPU': '1,2',
    # The total number of games when self-playing is:
    # Total = numSelfPlayProcess * numPerProcessSelfPlay
    'numSelfPlayProcess': 16,
    'numPerProcessSelfPlay': 50,
    # The total number of games when against-playing is:
    # Total = numAgainstPlayProcess * numPerProcessAgainst
    'numAgainstPlayProcess': 10,
    'numPerProcessAgainst': 10,
    'checkpoint': './temp/',
    'numItersForTrainExamplesHistory': 20,
    'lambdaHeur':0.0,
    'coeff':0.9,
    # Keep just the last N step of training, 0 if train from all steps
    'learnFromEnd':0,
})

if __name__=="__main__":
    import tensorflow as tf
    #tf.logging.info('TensorFlow')
    import logging
    log = logging.getLogger('tensorflow')
    fh = logging.FileHandler('tensorflow.log')
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    tf.logging.set_verbosity(tf.logging.ERROR)
    # tf.logging.info('TensorFlow')
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    g = Game(col=12, row=4, nir=7, defender=-1)
    c = Coach(g, args)
    c.learn()
