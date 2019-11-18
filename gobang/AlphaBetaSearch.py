import numpy as np
from gobang.GobangGame import display

class AlphaBetaSearch:
    def __init__(self, game):
        self.game = game
        self.maxdepth=5
        
    def play(self, board, curPlayer):
        valids = self.game.getValidMoves(board, 1)

    def search(self, board, curPlayer, depth, action):
        # ===== Description =====
        # Rerutns the average expected reward in the given
        #    (board, action, player) state
        next_s, next_player = self.game.getNextState(board, curPlayer, action)
        end =  self.game.getGameEnded(next_s, next_player, action)
        if(end !=0):
            #display(next_s, end = True)
            #print(curPlayer, end)
            return curPlayer
        if(depth == self.maxdepth):
            return 0
        else:
            valids = self.game.getValidMoves(next_s, next_player)
            result = []
            
            for act in range(len(valids)):
                if(valids[act]==0):
                    continue
                
                r = self.search(next_s, next_player, depth+1, act)
                if(r == next_player):
                    #display(next_s, end = True)
                    #print("www",r, next_player, depth)
                    return r
                else:
                    result.append(r)
            #print(result)
            return np.mean(result)
