import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
from pickle import Pickler, Unpickler
import os
from utils import *


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None, displaybar=True, mcts=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.displaybar = displaybar

        if(mcts!=None):
            self.mcts=mcts
            self.trainExamples=[]
            
    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        trainExamples=[]
        
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        action = -1
        while self.game.getGameEnded(board, curPlayer, action)==0:
            it+=1
            if verbose:
                assert(self.display)
                #print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)

            if(self.mcts!=None and self.mcts_player == curPlayer):
                action, data = self.mcts_play_and_collect_data(board, curPlayer)
                trainExamples.append(data)
            else:
                action = players[curPlayer+1](board, curPlayer)

            valids = self.game.getValidMoves(board,1)

            if valids[action]==0:
                #print(action)
                assert valids[action] >0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1, action)))
            self.display(board, end = True)

        if(self.mcts!=None):
            args=dotdict({'coeff': 0.9, 'learnFromEnd':0})
            
            mylist = []
            templist = []

            reward0 = self.game.getGameEnded(board, curPlayer, action)
            for i,x in enumerate(reversed(trainExamples[args.learnFromEnd:])):
                reward = (args.coeff**(i//2))*reward0*((-1)**(x[1]!=curPlayer))
                mylist.append((x[0], x[1], x[2], reward))
            templist.append(list(mylist))
            self.trainExamples.append(templist)
        return self.game.getGameEnded(board, 1, action), it

    def playGames(self, num, verbose=False, mcts = None):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
         a) In mcts mode:
            The trainExamples
         b) In normal mode:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
            
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        oneStepNum = 0.0
        twoStepNum = 0.0
        self.mcts_player=1
        for _ in range(num):
            gameResult, stepnum = self.playGame(verbose=verbose)
            oneStepNum+=stepnum
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            if(self.displaybar):
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=maxeps, et=eps_time.avg,
                                                                                                        total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()

        self.player1, self.player2 = self.player2, self.player1
        self.mcts_player=-1
        for _ in range(num):
            gameResult, stepnum = self.playGame(verbose=verbose)
            twoStepNum+=stepnum
            if gameResult==-1:
                oneWon+=1                
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            if(self.displaybar):
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num, et=eps_time.avg,
                                                                                                        total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()
            
        bar.finish()
        print(twoStepNum/(twoStepNum+oneStepNum))
        #self.log_data()

        if(self.mcts!=None):
            return self.trainExamples
        else:
            return oneWon, twoWon, twoStepNum/(twoStepNum+oneStepNum)

    def log_data(self):
        if(self.mcts!=None):
            #---save history---
            folder = './temp_try'
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, 'trainhistory.pth.tar'+".examples")
            with open(filename, "wb+") as f:
                Pickler(f).dump(self.trainExamples)
                f.closed

    def mcts_play_and_collect_data(self, board, curPlayer):
        pi, counts = self.mcts.getActionProb(board, curPlayer=curPlayer, debug=True)
        action = np.random.choice(len(pi), p=pi)
        mtx = self.mcts.heuristic.get_field_stregth_mtx(board, 1)
        heuristic_components = self.mcts.heuristic.get_x_line_mtx(board, 1)
        shape = list(board.shape)+[1]
        return action, [np.concatenate([np.reshape(board, shape),
                                              np.reshape(mtx, shape),
                                              heuristic_components], axis=2),
                              curPlayer, pi, None]
            
