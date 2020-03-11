import time
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
        self.lineMatrix = None
        self.lineMatrixInField = {}
        self.lineSize = None
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
            max = np.max(probs)
            probs[probs<max]=0
            probs /= np.sum(probs)
            a = np.random.choice(range(len(probs)), p = probs)
            return a

    def empty_line(self, line, board):
        for x0,y0 in line:
            if(board[x0][y0]==-1):
                return False
        return True

    def no_free_line(self, board):
        for key, lines in self.pointStrengthHeuristics.items():
            x, y = key
            if board[x][y]!=0:
                continue

            for line in lines:
                if self.empty_line(line, board):
                    return False

        return True
    
    def no_free_line2(self, board):
        enemy = np.logical_and(self.lineMatrix, board==-1)
        bad_lines = np.any(enemy, axis = (1,2))
        return np.all(bad_lines)
        
    def has_lost(self, board, player, action):
        x = action//self.N
        y = action % self.N

        for line in self.pointStrengthHeuristics[(x,y)]:
            is_full = True
            for x0,y0 in line:
                if(board[x0][y0]!=-player):
                    is_full = False
            if(is_full):
                return True
        return False
    
    def has_lost2(self, board, player, action):
        x = action//self.N
        y = action % self.N

        active_lines = np.sum(np.multiply(self.lineMatrix, board), axis = (1,2))
        return  np.any(np.equal(active_lines , self.lineSize))

    def line_sum(self, board):
        sum = 0.0
        lineSize = []
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
                #sum += 3.2**(-emptyNum)
                sum += 2.0**(-emptyNum)
            lineSize.append(enemyLess*emptyNum)
        return sum
    
    def line_sum2(self, board):
        enemy = np.logical_and(self.lineMatrix, board==-1)
        bad_lines = np.any(enemy, axis = (1,2))
        active_lines = np.sum(np.multiply(self.lineMatrix, board), axis = (1,2))

        lineSize = (self.lineSize-active_lines)
        lineSize = 1.0/np.exp2(lineSize.astype(float))
        lineSize *= (1-bad_lines)
        
        return np.sum(lineSize)

    def greedy(self, board):
        for x in range(self.M):
            for y in range(self.N):
                if(board[x][y]==0):
                    return x,y
        print("Board is full!!!")
        exit()

    # === NUMPY, fast version ===
    def get_x_line_mtx_new(self, board, player):
        mtx = np.zeros(shape = (self.M, self.N, 7))
        field = (board!=0)

        enemy = np.logical_and(self.lineMatrix, board==-1)
        bad_lines = np.any(enemy, axis = (1,2))
        active_lines = np.sum(np.multiply(self.lineMatrix, board), axis = (1,2))

        lineSize = (self.lineSize-active_lines)
        lineSize *= (1-bad_lines)

        for layer in range(7):
            act_layer = lineSize==(layer+1)
            mtx[...,layer] = np.tensordot(act_layer.transpose(),self.lineMatrix, axes=([0],[0]))
            mtx[...,layer] *= (1-field)
        return mtx

    # === NUMPY, fast version ===
    def get_field_stregth_mtx_new(self, board, player):
        enemy = np.logical_and(self.lineMatrix, board==-1)
        bad_lines = np.any(enemy, axis = (1,2))
        active_lines = np.sum(np.multiply(self.lineMatrix, board), axis = (1,2))

        lineSize = (self.lineSize-active_lines)
        lineSize = 1.0/np.exp2(lineSize.astype(float))
        lineSize *= (1-bad_lines)

        mtx = np.tensordot(lineSize,self.lineMatrix, axes=([0],[0]))
        mtx*= (1-(board!=0).astype(int))

        if np.max(mtx) == 0.0:
            x,y = self.greedy(board)
            mtx[x][y]=1.0
        return mtx

    # === OLD, slow version ===
    def get_x_line_mtx(self, board, player):
        mtx = np.zeros(shape = (self.M, self.N, 7))
        for key, lines in self.pointStrengthHeuristics.items():
            x,y = key
            if(board[x][y]!=0):
                continue

            for line in lines:
                enemyless = True
                emptynum = 0
                for (x1,y1) in line:
                    if(board[x1][y1]==-1):
                        enemyless = False
                        break
                    elif(board[x1][y1] == 0):
                        emptynum +=1
                if(enemyless):
                    mtx[x][y][emptynum-1] += 1
        return mtx

    # === OLD, slow version ===
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
        self.lineMatrix = np.zeros((len(self.Lines), self.M, self.N))
        for i,line in enumerate(self.Lines):
            for x,y in line:
                if (x,y) not in self.pointStrengthHeuristics:
                    self.pointStrengthHeuristics[(x,y)]=[line]
                else:
                    self.pointStrengthHeuristics[(x,y)].append(line)
                self.lineMatrix[i][x][y]=1

        for (x,y) in self.pointStrengthHeuristics:
            init_mtx = np.zeros((len(self.pointStrengthHeuristics[(x,y)]), self.M, self.N))
            
            for i,line in enumerate(self.pointStrengthHeuristics[(x,y)]):
                for (x0, y0) in line:
                    init_mtx[i][x0][y0]=1
            
            self.lineMatrixInField[(x,y)] = init_mtx

        self.lineSize = np.sum((self.lineMatrix), axis = (1,2))

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

from gobang.tensorflow.NNet import NNetWrapper as NNet
class PolicyPlayer():
    def __init__(self, game):
        self.game = game
        self.heuristic = Heuristic(game)
        
        self.nnet = NNet(game)
        self.nnet.load_checkpoint('./temp/','best.pth.tar')
        
    def play(self, board, curPlayer):
        mtx = self.heuristic.get_field_stregth_mtx(board, 1)
        heuristic_components = self.heuristic.get_x_line_mtx(board, 1)
        shape = list(np.shape(board))+[1]
        new_board = np.concatenate([np.reshape(board,shape),
                                    np.reshape(mtx, shape),
                                    heuristic_components], axis=2)
        start = time.time()
        probs, v, logits, exp_val, sum_val, valids2 = self.nnet.predict(new_board,curPlayer)
        end1 = time.time()
        valids = self.game.getValidMoves(board, curPlayer)
        move = np.argmax(probs*valids)
        end2 = time.time()
        #print(end1-start, end2-end1)
        return np.argmax(probs*valids)
