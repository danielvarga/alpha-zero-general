import os
import numpy as np
from gobang.GobangGame import GobangGame, display
from pickle import Pickler, Unpickler
from gobang.tensorflow.NNet import NNetWrapper as NNet
from Coach import Coach
from utils import *
from Arena import Arena
from MCTS import MCTS
from gobang.GobangPlayers import *
from gobang.tensorflow.NNet import NNetWrapper as nn

args = dotdict({
    'numMCTSSims': 500,
    'cpuct': 3.5,
    'multiGPU': False,
    'setGPU': '1,2',
    'checkpoint': './temp/',
    'lambdaHeur':0.0,
    'learnFromEnd':0,
    'evaluationDepth':1,
})

def load_history(history_folder = "temp", iterInterval = (0,None)):
    # ===== load history file =====
    # Descriptions: containes the data collected in every Iteration
    modelFile = os.path.join(history_folder, "trainhistory.pth.tar")
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
    begin, end = iterInterval
    for i,e in enumerate(trainhistory[begin:end]):
        trainExamples.extend(np.array(e))
        if(i>0):
            break

    for i in range(5):
        allBoard, curPlayer, pi, action = trainExamples[i]
        x,y,channelNum = np.shape(allBoard)
        board = np.squeeze(np.split(allBoard, 9, axis=2)[0], axis = 2)
        pi = np.reshape(pi[:-1], (x, y)).transpose()
        action = np.argmax(pi)
        print("Learned policy:\n", pi)
        display(board, end = True)
        print(action)
    return trainExamples

def calc_accuracy(nnet, trainExamples):
    allBoards, curPlayers, pis, rewards = zip(*trainExamples)
    actions = np.argmax(pis, axis = 1)
    allBoards = np.array(allBoards)
    curPlayers = np.array(curPlayers)
    
    new_actions = np.argmax(nnet.predict(allBoards, curPlayers, multi=True), axis = 1)
    print(new_actions, actions)
    accuracy = float(np.sum(new_actions==actions))/len(trainExamples)
    print("The accuracy is : {} ".format(accuracy))

def check_capabilities(args, n1, game):
    mcts1 = MCTS(game, n1, args, lambdaHeur=args.lambdaHeur)
    n1p =  lambda b, p: np.argmax(mcts1.getActionProb(b, p, temp=0))
    heuristic = Heuristic(game).play
    policy = PolicyPlayer(game).play

    arena = Arena(policy, heuristic,  game, display=display)
    print(arena.playGames(10, verbose=False))
    arena = Arena(n1p, heuristic,  game, display=display)
    print(arena.playGames(20, verbose=False))
   
    
def train_from_scratch(g, interval):
    trainExamples = load_history("temp", (0,interval[1]))
    print("Length: ", len(trainExamples))
    nnet = NNet(g)
    nnet.train(trainExamples)
    trainExamples = load_history("temp", (interval[1]//2,interval[1]))
    nnet.train(trainExamples)
    #trainExamples = load_history("temp", (interval[0],interval[1]))
    #nnet.train(trainExamples)
    # === Predict) ===
    calc_accuracy(nnet, trainExamples[interval[0]:interval[1]])
    check_capabilities(args, nnet, g)

def save_act_model(g):
    nnet = NNet(g)
    nnet.load_checkpoint('./temp/','best.pth.tar')
    nnet.save_model('cpp_loadmodel/amoba_model.pb')
    print("Save done")
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    g = GobangGame(col=8, row=4, nir=7, defender=-1)
    train_from_scratch(g, (0,5))
    #save_act_model(g)
