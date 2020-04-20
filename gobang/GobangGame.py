from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
# from .GobangLogic import Board
from .GobangPlayers import Heuristic
import numpy as np
import math

# defender=1: white aims for draw
# defender=-1: black aims for draw

class GobangGame(Game):
    def __init__(self, col=11, row=4, nir=7, defender=-1):
        self.col = col
        self.row = row
        self.n_in_row = nir
        self.defender = defender
        self.heuristic = Heuristic(self)

    def getInitBoard(self):
        # return initial board (numpy board)
        b = np.zeros((self.col, self.row))
        return b

    def getBoardSize(self):
        # (a,b) tuple
        return (self.col, self.row)

    def getActionSize(self):
        # return number of actions
        return self.col * self.row

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        
        move = (int(action / self.row), action % self.row)
        board = self.execute_move(board, move, player)
        return (board, -player)

    
    def hasValidMoves(self, board):
        return np.sum(board == 0) > 0

    def execute_move(self, board, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        (x,y) = move
        assert board[x][y] == 0, "{},{}".format(x,y)
        board[x][y] = color
        return board

    
    # modified
    def getValidMoves(self, raw_board, player):
        return (0 == raw_board.flatten()).astype('int')

    # def getReward(self, board, winner):
    #     return winner + winner*self.emptyFields(board)/20.0

    def getValidMoves_comp(self, compressed_board):
        invalids = np.bitwise_or(compressed_board[0],compressed_board[1])
        valids =  np.invert(invalids)
        return np.unpackbits(valids).astype(int)
        
    def compress_board(self, raw_board):
        player1 = (raw_board == 1)
        player2 = (raw_board == -1)
        
        #assert size % 8 ==0 # If not, extend with zeros
        p1 = np.packbits(player1.flatten())
        p2 = np.packbits(player2.flatten())
        return np.array([p1,p2])

    def decompress(self, compressed_board):
        p1 = np.unpackbits(compressed_board[0]).astype(int)
        p2 = np.unpackbits(compressed_board[1]).astype(int)
        #print(p1, p2)
        return  p1-p2
        
    def getNextState_comp(self, compressed_board, curPlayer, action):
        if(curPlayer == 1):
            curr = 0; next = 1;
        else:
            curr = 1; next = 0;

        #print((compressed_board[curr]) & (1<<action))
        assert not np.any(compressed_board[curr][action//8] & (1<<(7-action%8)))
        assert not np.any(compressed_board[next][action//8] & (1<<(7-action%8)))

        compressed_board[curr][action//8] |= (1<<(7-action%8))
        return compressed_board, -curPlayer

    def getGameEnded(self, board, player, action):
        """
        Return the Reward for the current board:
        +/- 1         : if GAME OVER
        0             : if game is not over
        """
        if(action < 0):
            return 0        
        
        if self.heuristic.has_lost(board, -1, action):
           return player
        #elif(np.min(np.absolute(board))==1):
        elif(self.heuristic.no_free_line(board)):
            # ===== This is a stronger stop  condition, than =====
           return -player
        else:
           return 0

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def display(board, end = False):
    M,N = np.shape(board)[:2]

    print("\n"+" "*30)
    print("=== Gobang Game ===")
    print(" "*(max(M+6, 30)))
    
    # Print the column indexis
    print("  ", end = '')
    for i in range(M):
        print(str(i)[-1]+" ", end = '')
    print(' '*10)

    print(" +"+"=="*(M)+"+")
    for x in range(N):
        print("{}|".format(x), end = '')
        for y in range(M):
            piece = board[y][x]
            if(piece>0):
                print("{}{}{}".format(bcolors.WARNING, "o ", bcolors.ENDC), end = '')
            elif(piece<0):
                print("{}{}{}".format(bcolors.FAIL, "x ", bcolors.ENDC), end = '')
            else:
                print("  ", end = '')
        print("|")
    print((" +"+"=="*(M)+"+"))

    # Go back, if not finished yet
    print("\n", end = "")
    if(not end):
        print('\033[{}A'.format(N+9))
    else:
        print('')

#def display(board):
#    col = board.shape[0]
#    row = board.shape[1]
#
#    print("   |", end="")
#    for x in range(col):
#        print("{:3d}|".format(x), end="")
#    print("")
#    print(" -----------------------")
#    for y in range(row):
#        print("{:3d}|".format(y), end="")    # print the row #
#        for x in range(col):
#            piece = board[x][y]    # get the piece to print
#            if piece == -1:
#                print(" b  ", end="")
#            elif piece == 1:
#                print(" W  ", end="")
#            else:
#                print(" -  ", end="")
#                    
#        print("|")
#
#    print("   -----------------------")
