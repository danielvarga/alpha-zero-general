import os
import numpy as np
from gobang.GobangGame import GobangGame, display
from pickle import Pickler, Unpickler
from gobang.tensorflow.NNet import NNetWrapper as NNet

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

    for i in range(10):
        allBoard, curPlayer, pi, action = trainExamples[i]
        x,y,channelNum = np.shape(allBoard)
        board = np.squeeze(np.split(allBoard, 9, axis=2)[0], axis = 2)
        pi = np.reshape(pi[:-1], (x, y)).transpose()
        print("Learned policy:\n", pi)
        display(board, end = True)
        print(action)
    return trainExamples

def calc_accuracy(nnet, trainExamples):
    allBoards = np.array([t[0] for t in trainExamples])
    curPlayers = np.array([t[1] for t in trainExamples])
    actions = [np.random.choice(len(t[2]), p=t[2]) for t in trainExamples]
    rewards = np.array([t[3] for t in trainExamples])

    new_actions = np.argmax(nnet.predict(allBoards, curPlayers, multi=True), axis = 1)
    print(new_actions)
    print(actions)
    accuracy = float(np.sum(new_actions==actions))/len(trainExamples)
    print("The accuracy is : {} ".format(accuracy))
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    g = GobangGame(col=12, row=4, nir=7, defender=-1)

    trainExamples = load_history("temp", (0,1))
    nnet = NNet(g)
    nnet.train(trainExamples)
    # === Predict) ===
    calc_accuracy(nnet, trainExamples[:10])
    print(len(trainExamples))

    
