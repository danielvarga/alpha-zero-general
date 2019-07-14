from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .GobangLogic import Board
import numpy as np


# defender=1: white aims for draw
# defender=-1: black aims for draw
class GobangGame(Game):
    def __init__(self, col=11, row=4, nir=7, defender=-1):
        self.col = col
        self.row = row
        self.n_in_row = nir
        self.defender = defender

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.col, self.row)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.col, self.row)

    def getActionSize(self):
        # return number of actions
        return self.col * self.row + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.col * self.row:
            return (board, -player)
        b = Board(self.col, self.row)
        b.pieces = np.copy(board)
        move = (int(action / self.row), action % self.row)
        b.execute_move(move, player)
        return (b.pieces, -player)

    # modified
    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board(self.col, self.row)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.row * x + y] = 1
        return np.array(valids)

    # modified
    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player won, -1 if player lost
        # player = 1
        b = Board(self.col, self.row)
        b.pieces = np.copy(board)
        n = self.n_in_row
        col = self.col
        row = self.row

        # display(board)

        for w in range(col):
            # if the offence has a full column, he won
            if set(board[w][i] for i in range(self.row)) == {-self.defender}:
                return -self.defender
            
            # if the offence has a row of len self.n_in_row, he won
            if (w in range(row - n + 1)):
                for h in range(row):
                    if set(board[i][h] for i in range(w, w + n)) == {-self.defender}:
                        return -self.defender

            # if the offence has a full diagonal of length col, he won
            if (w in range(col - row + 1)):
                if set(board[w+l][l] for l in range(row)) == {-self.defender}:
                    return -self.defender
            if (w in range(row-1, col)):
                if set(board[w-l][l] for l in range(row)) == {-self.defender}:
                    return -self.defender
            
                    
            # if the offence has 3 in a corner diagonal, he won
            if set((board[2][0], board[1][1], board[0][2])) == {-self.defender}:
                return -self.defender
            if set((board[col-3][0], board[col-2][1], board[col-1][2])) == {-self.defender}:
                return -self.defender
            if set((board[0][row-3], board[1][row-2], board[2][row-1])) == {-self.defender}:
                return -self.defender
            if set((board[col-1][row-3], board[col-2][row-2], board[col-3][row-1])) == {-self.defender}:
                return -self.defender

            # if the offence has two in one of the northern corner diagonals, he won
            if set((board[1][0], board[0][1])) == {-self.defender}:
                return -self.defender
            if set((board[col-2][0], board[col-1][1])) == {-self.defender}:
                return -self.defender
            
        if b.has_legal_moves():
            return 0

        # game is over and the defender has won
        return self.defender

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player * board

    # modified
    def getSymmetries(self, board, pi):
        assert False
        # mirror, rotational
        assert(len(pi) == self.n**2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()


def display(board):
    col = board.shape[0]
    row = board.shape[1]

    print("   |", end="")
    for x in range(col):
        print("{:3d}|".format(x), end="")
    print("")
    print(" -----------------------")
    for y in range(row):
        print("{:3d}|".format(y), end="")    # print the row #
        for x in range(col):
            piece = board[x][y]    # get the piece to print
            if piece == -1:
                print(" b  ", end="")
            elif piece == 1:
                print(" W  ", end="")
            else:
                print(" -  ", end="")
                    
        print("|")

    print("   -----------------------")
