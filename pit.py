import Arena
from MCTS import MCTS
from gobang.GobangGame import GobangGame, display
from gobang.GobangPlayers import *
from gobang.tensorflow.NNet import NNetWrapper as NNet
import os
import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import multiprocessing
from utils import *
from pytorch_classification.utils import Bar, AverageMeter

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

def Async_Play(game,args,iter_num,bar):
    bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(i=iter_num+1,x=args.numPlayGames,total=bar.elapsed_td, eta=bar.eta_td)
    bar.next()

    # set gpu
    if(args.multiGPU):
        if(iter_num%2==0):
            os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    # set gpu growth
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    sess = tf.Session(config=config)

    # create NN
    model1 = NNet(game)
    model2 = NNet(game)

    # try load weight
    try:
        model1.load_checkpoint(folder=args.model1Folder, filename=args.model1FileName)
    except:
        print("load model1 fail")
        pass
    try:
        model2.load_checkpoint(folder=args.model2Folder, filename=args.model2FileName)
    except:
        print("load model2 fail")
        pass

    # create MCTS
    mcts1 = MCTS(game, model1, args)
    mcts2 = MCTS(game, model2, args)

    # each process play 2 games
    arena = Arena.Arena(lambda x: np.argmax(mcts1.getActionProb(x, temp=0)),lambda x: np.argmax(mcts2.getActionProb(x, temp=0)), game)
    arena.displayBar = False
    oneWon,twoWon, draws = arena.playGames(2)
    return oneWon,twoWon, draws

if __name__=="__main__":
    """
    Before using multiprocessing, please check 2 things before use this script.
    1. The number of PlayPool should not over your CPU's core number.
    2. Make sure all Neural Network which each process created can store in VRAM at same time. Check your NN size before use this.
    """

    parser = argparse.ArgumentParser(description='Struggle your Models with each other')
    parser.add_argument('-mode', dest = 'mode', help='Choose the mode: tournament, human, one2one')
    modeargs = parser.parse_args()

    args = dotdict({
    'numMCTSSims': 25,
    'cpuct': 3,

    'multiGPU': True,  # multiGPU only support 2 GPUs.
    'setGPU': '4,5',
    'numPlayGames': 4,  # total num should x2, because each process play 2 games.
    'numPlayPool': 4,   # num of processes pool.

    'model1Folder': './temp/',
    'model1FileName': 'best.pth.tar',
    'model2Folder': './temp/',
    'model2FileName': 'best.pth.tar',

    })

    def ParallelPlay(g):
        bar = Bar('Play', max=args.numPlayGames)
        pool = multiprocessing.Pool(processes=args.numPlayPool)
        res = []
        result = []
        for i in range(args.numPlayGames):
            res.append(pool.apply_async(Async_Play,args=(g,args,i,bar)))
        pool.close()
        pool.join()

        oneWon = 0
        twoWon = 0
        draws = 0
        for i in res:
            result.append(i.get())
        for i in result:
            oneWon += i[0]
            twoWon += i[1]
            draws += i[2]
        print("Model 1 Win:",oneWon," Model 2 Win:",twoWon," Draw:",draws)


    g = GobangGame(col=8, row=4, nir=7, defender=-1)

    # parallel version
    #ParallelPlay(g)

    # single process version
    # all players
    rp = RandomPlayer(g).play
    hp = HumanGobangPlayer(g).play
    heuristic = Heuristic(g).play

    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./temp/','best.pth.tar')

    args1 = dotdict({'numMCTSSims': 50, 'cpuct':3.0, 'multiGPU':False})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda b, p: np.argmax(mcts1.getActionProb(b, p, temp=0))

    all = {
        'Random': rp,
        'Best NN': n1p,
        'Heuristic': heuristic,
    }
    if modeargs.mode == 'human':
        arena = Arena.Arena(n1p, hp, g, display=display)
        print(arena.playGames(2, verbose=True))
    elif modeargs.mode == 'one2one':
        arena = Arena.Arena(rp, heuristic, g, display=display)
        print(arena.playGames(2, verbose=True))
    elif modeargs.mode == 'one2all':
        results = []
        y = []
        for name,player in all.items():
            arena = Arena.Arena(player,rp, g, display=display)
            firstWin, secondWin, draw = arena.playGames(2, verbose=False)
            rate = (firstWin*3 +draw)/ (3*firstWin+3*secondWin+3*draw)
            results.append(rate)
            y.append(name)
        plt.plot(y, results)
        plt.show()
    elif modeargs.mode == 'tournament':
        results = [[""]+[name for name,p in all.items()]]
        for name1,player1 in all.items():
            row = [name1]
            for name2,player2 in all.items():
                arena = Arena.Arena(player1, player2, g, display=display)
                firstWin, secondWin, draw = arena.playGames(2, verbose=False)
                rate = (firstWin*3 +draw)/ (3*firstWin+3*secondWin+3*draw)
                row.append(rate)
                print(name1, name2,rate)
            results.append(row)
        for row in results:
            for item in row:
                print("{:>10}".format(item), end = '')
            print('')
    else:
        print('Mode not found: {}'.format(modeargs.mode))
