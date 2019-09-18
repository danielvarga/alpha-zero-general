from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os
from pickle import Pickler, Unpickler
import tensorflow as tf
import multiprocessing
from gobang.tensorflow.NNet import NNetWrapper as nn
from gobang.GobangGame import display

def AsyncSelfPlay(game,args,iter_num,bar):
    #set gpu
    if(args.multiGPU):
        if(iter_num%2==0):
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    #set gpu memory grow
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    sess = tf.Session(config=config)

    #create nn and load weight
    net = nn(game, args.displaybar)
    try:
        net.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
    except:
        pass
    mcts = MCTS(game, net,args)

    # create separate seeds for each worker
    np.random.seed(iter_num)
   
    # create a list for store game state
    returnlist = []
    for i in range(args.numPerProcessSelfPlay):
        # Each process play many games, so do not need initial NN every times when process created.

        if args.displaybar:
            bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(i=i+1,x=args.numPerProcessSelfPlay,total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        trainExamples = []
        board = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        while True:
            templist = []
            episodeStep += 1
            temp = int(episodeStep < args.tempThreshold)

            pi = mcts.getActionProb(board, curPlayer=curPlayer, temp=temp)
            trainExamples.append([board, curPlayer, pi, None])
            # sym = game.getSymmetries(board, pi)
            # for b,p in sym:
            #     trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = game.getNextState(board, curPlayer, action)

            r = game.getGameEnded(board, curPlayer)

            if r!=0: # game is over
                templist.append(list((x[0], x[1], x[2], r*((-1)**(x[1]!=1))) for x in trainExamples))
                returnlist.append(templist)
                break

    return returnlist

def AsyncTrainNetwork(game,args,trainhistory):

    #set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU
    #create network for training
    nnet = nn(game, args.displaybar)
    try:
        nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
    except:
        pass
    #---load history file---
    modelFile = os.path.join(args.checkpoint, "trainhistory.pth.tar")
    examplesFile = modelFile+".examples"
    if not os.path.isfile(examplesFile):
        print(examplesFile)
    else:
        print("File with trainExamples found. Read it.")
        with open(examplesFile, "rb") as f:
            for i in Unpickler(f).load():
                trainhistory.append(i)
        f.closed
    #----------------------
    #---delete if over limit---
    if len(trainhistory) > args.numItersForTrainExamplesHistory:
        print("len(trainExamplesHistory) =", len(trainhistory), " => remove the oldest trainExamples")
        del trainhistory[len(trainhistory)-1]
    #-------------------
    #---extend history---
    trainExamples = []
    for e in trainhistory:
        trainExamples.extend(e)
    #---save history---
    folder = args.checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, 'trainhistory.pth.tar'+".examples")
    with open(filename, "wb+") as f:
        Pickler(f).dump(trainhistory)
        f.closed
    #------------------
    nnet.train(trainExamples)
    nnet.save_checkpoint(folder=args.checkpoint, filename='train.pth.tar')

def AsyncAgainst(game,args,iter_num,bar):
    # create separate seeds for each worker
    # np.random.seed(iter_num)

    if args.displaybar:
        bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(i=iter_num+1,x=args.numAgainstPlayProcess,total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()

    #set gpu
    if(args.multiGPU):
        if(iter_num%2==0):
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    #set gpu memory grow
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    sess = tf.Session(config=config)
              
    #create nn and load
    nnet = nn(game, args.displaybar)
    pnet = nn(game, args.displaybar)
    try:
        nnet.load_checkpoint(folder=args.checkpoint, filename='train.pth.tar')
    except:
        print("load train model fail")
        pass
    try:
        pnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
    except:
        print("load old model fail")
        filepath = os.path.join(args.checkpoint, "best.pth.tar")
        pnet.save_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
    pmcts = MCTS(game, pnet, args)
    nmcts = MCTS(game, nnet, args)

    arena = Arena(lambda b, p: np.argmax(pmcts.getActionProb(canonicalBoard=b, curPlayer=p, temp=1)),
                    lambda b, p: np.argmax(nmcts.getActionProb(canonicalBoard=b, curPlayer=p, temp=1)), game, displaybar=args.displaybar)
    # each against process play the number of numPerProcessAgainst games.
    pwins, nwins, draws = arena.playGames(args.numPerProcessAgainst)
    return pwins, nwins, draws

def CheckResultAndSaveNetwork(pwins,nwins,draws,game,args,iter_num):
    #set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    if pwins+nwins > 0 and float(nwins+(0.5*draws))/(pwins+nwins+draws) < args.updateThreshold:
        print('REJECTING NEW MODEL')
    else:
        print('ACCEPTING NEW MODEL')
        net = nn(game, args.displaybar)
        net.load_checkpoint(folder=args.checkpoint, filename='train.pth.tar')
        net.save_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
        net.save_checkpoint(folder=args.checkpoint, filename='checkpoint_' + str(iter_num) + '.pth.tar')

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.trainExamplesHistory = []

    def parallel_self_play(self):
        temp = []
        result = []
        bar = Bar('Self Play(each process)', max=self.args.numPerProcessSelfPlay)
        if self.args.multiCPU:
            pool = multiprocessing.Pool(processes=self.args.numSelfPlayProcess)
            res = []
            for i in range(self.args.numSelfPlayProcess):
                res.append(pool.apply_async(AsyncSelfPlay,args=(self.game,self.args,i,bar,)))
            pool.close()
            pool.join()
            for i in res:
                result.append(i.get())
        else:
            result.append(AsyncSelfPlay(self.game, self.args, 0, bar))
            
        for i in result:
            for j in i:
                for trainData in j:
                    temp += trainData
        return temp

    def parallel_train_network(self,iter_num):
        print("Start train network")
        if self.args.multiCPU:
            pool = multiprocessing.Pool(processes=1)
            pool.apply_async(AsyncTrainNetwork,args=(self.game,self.args,self.trainExamplesHistory,))
            pool.close()
            pool.join()
        else:
            AsyncTrainNetwork(self.game, self.args, self.trainExamplesHistory)

    def parallel_self_test_play(self,iter_num):
        print("Start test play")
        bar = Bar('Test Play', max=self.args.numAgainstPlayProcess)
        result = []
        if self.args.multiCPU:
            pool = multiprocessing.Pool(processes=self.args.numAgainstPlayProcess)
            res = []
            for i in range(self.args.numAgainstPlayProcess):
                res.append(pool.apply_async(AsyncAgainst,args=(self.game,self.args,i,bar)))
            pool.close()
            pool.join()
            for i in res:
                result.append(i.get())
        else:
            result.append(AsyncAgainst(self.game, self.args, 0, bar))

        pwins = 0
        nwins = 0
        draws = 0
        for i in result:
            pwins += i[0]
            nwins += i[1]
            draws += i[2]

        print("pwin: "+str(pwins))
        print("nwin: "+str(nwins))
        print("draw: "+str(draws))
        if self.args.multiCPU:
            pool = multiprocessing.Pool(processes=1)
            pool.apply_async(CheckResultAndSaveNetwork,args=(pwins,nwins,draws,self.game,self.args,iter_num,))
            pool.close()
            pool.join()
        else:
            CheckResultAndSaveNetwork(pwins, nwins, draws, self.game, self.args, iter_num)

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            print('------ITER ' + str(i) + '------')
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            temp = self.parallel_self_play()
            iterationTrainExamples += temp
            self.trainExamplesHistory.append(iterationTrainExamples)
            self.parallel_train_network(i)
            self.trainExamplesHistory.clear()
            self.parallel_self_test_play(i)
