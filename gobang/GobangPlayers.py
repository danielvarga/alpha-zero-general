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

class Heuristic():
    def __init__(self, game):
        self.game = game
        self.N = game.row
        self.M = game.col
        self.generate_lines()

    def play(self, board, curPlayer):
        if(curPlayer != 0):
            mtx = self.get_field_stregth_mtx(board, 1)
            x,y = np.unravel_index(np.argmax(mtx), mtx.shape)
            if(board[x][y]!=0):
                x,y = self.greedy(board)
            #print(x, y, mtx[x][y])
            return x*self.N+y
        else:
            print("Invalid player!!!", curPlayer)

    def random_play(self, board, curPlayer):
        if(curPlayer != 0):
            mtx = self.get_field_stregth_mtx(board, 1)
            probs = np.array(mtx).flatten()
            a = np.random.choice(range(len(probs)), p = probs)
            return a
        
    def line_sum(self, board):
        sum = 0.0
        for line in self.Lines:
            enemyLess = True
            emptyNum = 0
            for x,y in line:
                if(board[x][y]==-1):
                    enemyLess = False
                    break;
                elif(board[x][y]==0):
                    emptyNum += 1
            if(enemyLess):
                sum += 2.0**(-emptyNum)
        return sum
             
    def greedy(self, board):
        for x in range(self.M):
            for y in range(self.N):
                if(board[x][y]==0):
                    return x,y
        print("Board is full!!!")
        exit()
        
    def get_field_stregth_mtx(self, board, player, verbose=False):
        mtx = np.zeros(shape = (self.M, self.N))
        for key, lines in self.pointStrengthHeuristics.items():
            x,y = key
            if(board[x][y]!=0):
                continue

            min_emptynum = 100
            for line in lines:
                enemyless = True
                emptynum = 0
                for (x1,y1) in line:
                    if(board[x1][y1]==-player):
                        enemyless = False
                        break
                    elif(board[x1][y1] == 0):
                        emptynum +=1
                if(enemyless):
                    min_emptynum = min(emptynum, min_emptynum)
                    mtx[x][y] += 2.0**(-emptynum)

        #mtx = mtx**2
        if verbose:
            mtx = mtx.transpose()
            for row in mtx:
                for x in row:
                    print("{0:.4f}".format(x), end = ' ')
                print('')

            for row in board:
                for x in row:
                    print("{}".format(x), end = ' ')
                print("")
            mtx = mtx.transpose()

        if np.max(mtx) == 0.0:
            x,y = self.greedy(board)
            mtx[x][y]=1.0
        return mtx

       
    def generate_lines(self, player = 1):
        self.Lines = []
        self.pointStrengthHeuristics={}
        col, row = (self.M, self.N)
        n = 7

        for w in range(col):
            # if the offence has a full column, he won
            line = [(w, i) for i in range(row)]
            self.Lines.append(line)

            # if the offence has a row of len self.n_in_row, he won
            if (w in range(1, col - n)):
                for h in range(row):
                    line = [(i,h) for i in range(w, w + n)]
                    self.Lines.append(line)

            # if the offence has a row of len 4 in the border, he won
            #half = int(math.ceil(n/2))
            half = 4
            if (w in [0,col-half]):
                for h in range(row):
                    line = [(i,h) for i in range(w, w + half)]
                    self.Lines.append(line)

            # if the offence has a full diagonal of length col, he won
            if (w in range(col - row + 1)):
                line = [(w+l,l) for l in range(row)]
                self.Lines.append(line)
            if (w in range(row-1, col)):
                line = [(w-l,l) for l in range(row)]
                self.Lines.append(line)

        # if the offence has 3 in a corner diagonal, he won
        self.Lines.append([(2,0),(1,1), (0,2)])
        self.Lines.append([(col-3,0),(col-2,1), (col-1,2)])
        self.Lines.append([(0, row-3),(1, row-2), (2, row-1)])
        self.Lines.append([(col-1, row-3),(col-2, row-2), (col-3, row-1)])

        # if the offence has two in one of the northern corner diagonals, he won
        self.Lines.append([(1,0),(0,1)])
        self.Lines.append([(col-2,0),(col-1,1)])


        # Geather the lines in each Field
        for line in self.Lines:
            for x,y in line:
                if (x,y) not in self.pointStrengthHeuristics:
                    self.pointStrengthHeuristics[(x,y)]=[line]
                else:
                    self.pointStrengthHeuristics[(x,y)].append(line)

class HumanGobangPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, curPlayer):
        # display(board)
        verbose = False
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                if verbose:
                    print("({} {}) ".format(int(i/self.game.row), int(i%self.game.row)), end="")
        if verbose: print("")
        while True:
            a = input()

            try:
                x,y = [int(x) for x in a.split(' ')]
            except:
                print("Bad value, try again")
                a = input()
                x,y = [int(x) for x in a.split(' ')]
            a = self.game.row * x + y if x!= -1 else self.game.row * self.game.col
            if valid[a]:
                break
            else:
                x = int(a/self.game.row)
                y = int(a%self.game.row)
                print('\033[1A{} {}      Invalid '.format(x,y), end = '')
                #print(" "*30+'\r', end = '')

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
