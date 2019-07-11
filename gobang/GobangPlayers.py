import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, curPlayer):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanGobangPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, curPlayer):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("({} {}) ".format(int(i/self.game.row), int(i%self.game.row)), end="")
        print("")
        while True:
            a = input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.row * x + y if x!= -1 else self.game.row * self.game.col
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class GreedyGobangPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, curPlayer):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
