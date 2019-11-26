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
    'numIters': 30,
    'numEps': 100,
    'tempThreshold': 48,
    'updateThreshold': 0.51,
    'maxlenOfQueue': 320000,
    'numMCTSSims': 400,
    'cpuct': 3.5,
    'multiGPU': True,
    'setGPU': '1,2',
    # The total number of games when self-playing is:
    # Total = numSelfPlayProcess * numPerProcessSelfPlay
    'numSelfPlayProcess': 20,
    'numPerProcessSelfPlay': 40,
    # The total number of games when against-playing is:
    # Total = numAgainstPlayProcess * numPerProcessAgainst
    'numAgainstPlayProcess': 20,
    'numPerProcessAgainst': 4,
    'checkpoint': './temp/',
    'numItersForTrainExamplesHistory': 20,
    'lambdaHeur':0.0,
    'coeff':0.9,
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

    g = Game(col=8, row=4, nir=7, defender=-1)
    c = Coach(g, args)
    c.learn()
