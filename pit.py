import Arena
from MCTS import MCTS
from gobang.GobangGame import GobangGame, display
from gobang.GobangPlayers import *
from gobang.tensorflow.NNet import NNetWrapper as NNet
import os
import random
import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import multiprocessing
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
import matplotlib
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

def myheatmap(data, xlabels, ylabels):
  fig, ax = plt.subplots()
  im = plt.imshow(data.transpose(), cmap='RdYlGn')
  cbar = ax.figure.colorbar(im, ax=ax)
  ax.set_xticks(np.arange(data.shape[0]))
  ax.set_yticks(np.arange(data.shape[1]))
  ax.set_xticklabels(xlabels)
  ax.set_yticklabels(ylabels)
  plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
               rotation_mode="anchor")

  valfmt = matplotlib.ticker.StrMethodFormatter("{x:.1f} %")
  kw = dict(horizontalalignment="center",
                verticalalignment="center")
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      text = im.axes.text(i, j, valfmt(100*data[i][j]), **kw)

  plt.title("Heatmap")
  fig.savefig("heatmap.png")
  plt.show()

def Async_Play(game,args,iter_num,bar):
    bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(
        i=iter_num+1,x=args.numPlayGames,total=bar.elapsed_td, eta=bar.eta_td)
    bar.next()

    # set gpu
    if(args.multiGPU):
        if(iter_num%2==0):
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    arena = Arena.Arena(lambda x: np.argmax(mcts1.getActionProb(x, temp=0)),
                        lambda x: np.argmax(mcts2.getActionProb(x, temp=0)), game)
    arena.displayBar = False
    oneWon,twoWon, draws = arena.playGames(2)
    return oneWon,twoWon, draws

if __name__=="__main__":
    """
    Before using multiprocessing, please check 2 things before use this script.
    1. The number of PlayPool should not over your CPU's core number.
    2. Make sure all Neural Network which each process created can store 
    in VRAM at same time. Check your NN size before use this.
    """

    parser = argparse.ArgumentParser(description='Struggle your Models with each other')
    parser.add_argument('-mode', dest = 'mode', help='Choose the mode: tournament, human, one2one')
    parser.add_argument('-gpu', dest = 'gpu', help='Select a free GPU')
    modeargs = parser.parse_args()

    args = dotdict({
    'numMCTSSims': 25,
    'cpuct': 3,

    'multiGPU': False,  # multiGPU only support 2 GPUs.
    'setGPU': '1,2',
    'numPlayGames': 4,  # total num should x2, because each process play 2 games.
    'numPlayPool': 4,   # num of processes pool.

    'model1Folder': './temp/',
    'model1FileName': 'best.pth.tar',
    'model2Folder': './temp/',
    'model2FileName': 'best.pth.tar',
    'battleNum': 20,
    'lambdaHeur': 1.0
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


    g = GobangGame(col=12, row=4, nir=7, defender=-1)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = modeargs.gpu
    # parallel version
    #ParallelPlay(g)

    # single process version
    # all players
    rp = RandomPlayer(g).play
    hp = HumanGobangPlayer(g).play
    heuristic_rand = Heuristic(g).random_play
    heuristic = Heuristic(g).play
    policyPlayer = PolicyPlayer(g).play
    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./temp/','best.pth.tar')

    args1 = dotdict({'numMCTSSims': 400, 'cpuct':3.5, 'evaluationDepth':1, 'multiGPU': args.multiGPU, 'setGPU':args.setGPU,'alpha':0.3,'epsilon':0.25,})
    mcts1 = MCTS(g, n1, args1, lambdaHeur=0.1)
    n1p = lambda b, p: np.argmax(mcts1.getActionProb(b, p, temp=0))

    # improved nnet player
    n2 = NNet(g)
    n2.load_checkpoint('./temp/','best.pth.tar')
    args2 = dotdict({'numMCTSSims':1000, 'cpuct':0.1, 'multiGPU':True,'alpha':0.3,'epsilon':0.25,})
    mcts2 = MCTS(g, n2, args2, lambdaHeur=1.0)
    n2p =  lambda b, p: np.argmax(mcts2.getActionProb(b, p, temp=0))

      
    all = {
        'Random': rp,
        'Best NN': n1p,
        'Heuristic': heuristic,
        '$\lambda$-Heur-{}'.format(args.lambdaHeur):n2p,
    }
    
    if modeargs.mode == 'human':
        arena = Arena.Arena(policyPlayer, hp, g, display=display)
        print(arena.playGames(4, verbose=True))
    elif modeargs.mode == 'one2one':
        arena = Arena.Arena(policyPlayer, heuristic,  g, display=display)
        print(arena.playGames(100, verbose=True))
    elif modeargs.mode == 'one2all':
        results = []
        y = []
        for name,player in all.items():
            arena = Arena.Arena(player, n2p, g, display=display)
            firstWin, secondWin, draw = arena.playGames(args.battleNum, verbose=False)
            rate = (firstWin*3 +draw)/ (3*firstWin+3*secondWin+2*draw)
            results.append(rate)
            y.append(name)
        plt.plot(y, results)
        plt.show()
    elif modeargs.mode == 'tournament':
        #results = [[""]+[name for name,p in all.items()]]
        results= []
        for name1,player1 in all1.items():
            #row = [name1]
            row = []
            for name2,player2 in all2.items():
                arena = Arena.Arena(player1, player2, g, display=display)
                firstWin, secondWin, draw = arena.playGames(20, verbose=False)
                rate = (firstWin)/ (firstWin+secondWin)
                row.append(rate)
                print(name1, name2,rate, firstWin, secondWin, draw)
            results.append(row)
        #for row in results:
        #    for item in row:
        #        print("{:>15}".format(item), end = '')
        #    print('')
        labels1 = [name for name,p in all1.items()]
        labels2 = [name for name,p in all2.items()]
        myheatmap(np.array(results), labels1, labels2)
    else:
        print('Mode not found: {}'.format(modeargs.mode))
