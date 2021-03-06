import math
import numpy as np
EPS = 1e-8

from gobang.GobangGame import display
from gobang.GobangPlayers import Heuristic

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, lambdaHeur=0.0):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s
        self.heuristic = Heuristic(game)
        self.lambdaHeur = lambdaHeur
        self.alphas = [args.alpha]*game.getActionSize()
        self.action_size = game.getActionSize()
        self.depth_count = np.zeros(self.action_size+1).astype(int)
        
    def getActionProb(self, board, curPlayer, temp=1, debug=False):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        self.endNum = 0
        self.win = 0

        if self.args.fast_eval:
            for i in range(self.args.numMCTSSims):
                tmp_board = np.copy(board)
                self.fast_search(tmp_board, curPlayer, -1)

                #s = board.tobytes()
                s = self.game.compress_board(board).tobytes()
                counts = self.Nsa[s]
        else:
            for i in range(self.args.numMCTSSims):
                tmp_board = np.copy(board)
                self.search(tmp_board, curPlayer, -1)

                #s = board.tobytes()
                s = self.game.compress_board(board).tobytes()
                counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
                #counts = np.array([self.Qsa[(s,a)] if (s,a) in self.Qsa else 0 for a in range(self.game.getActionSize())])
        
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            if debug:
                return probs, counts
            else:
                return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        if debug:
            return probs, counts
        else:
            return probs

    def fast_search(self, raw_board, curPlayer, action):
        #s = raw_board.tobytes()
        s = self.game.compress_board(raw_board).tobytes()

        
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(raw_board, curPlayer, action)
        if self.Es[s]!=0:
            self.endNum+=1
            self.win += (curPlayer == 1)
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            self.depth_count[int(np.sum(np.abs(raw_board)))]+=1
            
            # leaf node
            # === get v by simulation ===:
            #v = self.fast_search(raw_board, curPlayer, action)
            valids = self.game.getValidMoves(raw_board, curPlayer)

            mtx = self.heuristic.get_field_stregth_mtx(raw_board, 1)
            probs = mtx.flatten()/np.sum(mtx)

            # === Add Dirichlet noise ===
            #eps = self.args.epsilon
            eps = 0.1
            noise = np.random.dirichlet(self.alphas)
            probs = eps*noise+(1.0-eps)*probs

            self.Ps[s] = probs
            self.Vs[s] = valids
            self.Ns[s] = 0
            self.Nsa[s] = np.zeros(self.action_size)
            self.Qsa[s] = np.zeros(self.action_size)
            # TODO:
            #    --> fast search here???
            #    --> choose action radom/heuristic ...
            #    -->
            v = curPlayer*self.get_heuristic_end(raw_board, curPlayer, action)
            return -v
            
        valids = self.Vs[s]

        u = self.Qsa[s]+self.args.cpuct*self.Ps[s]*math.sqrt(self.Ns[s]+EPS)/(1+self.Nsa[s])
        a = np.argmax((u+2)*valids);

        #print(raw_board)
        #print(u, q, p, n)
        #print(a, valids)
        
        next_s, next_player = self.game.getNextState(raw_board, curPlayer, a)

        v = self.fast_search(next_s, next_player, a)
        Nsa = self.Nsa[s][a]
        self.Qsa[s][a] = (Nsa*self.Qsa[s][a] + v)/(Nsa+1)
        self.Nsa[s][a] += 1

        self.Ns[s] += 1
        return -v
        
    def search(self, raw_board, curPlayer, action):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = raw_board.tobytes()
        #display(canonicalBoard, end = True)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(raw_board, curPlayer, action)
        if self.Es[s]!=0:
            self.endNum+=1
            self.win += (curPlayer == 1)
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            mtx = self.heuristic.get_field_stregth_mtx(raw_board, 1)
            heuristic_components = self.heuristic.get_x_line_mtx(raw_board, 1)

            # === If next step is a obligatory ===
            oneStepWin = (heuristic_components[...,0]>0).any()
            twoStepWin = (heuristic_components[...,1]>1).any()
            if oneStepWin or twoStepWin:
                # Move the only choice:
                chanel = 0 if oneStepWin else 1
                
                a = np.argmax(heuristic_components[...,chanel].flatten())
                onehot = np.zeros((self.game.getActionSize(),))
                onehot[a] = 1
                self.Vs[s]= onehot
                self.Ns[s]=0

                next_s, next_player = self.game.getNextState(raw_board, curPlayer, a)
                v = self.search(next_s, next_player, a)
                self.Qsa[(s,a)] = v
                self.Nsa[(s,a)] = 1
                
                return -v
            
            elif self.args.evaluationDepth > 1:
                probs, v = self.evalSituation(raw_board, curPlayer, action)
                v = v[0]
                #print("Value after {} eval: {}".format(self.args.evaluationDepth,-v))
            else:
                shape = list(raw_board.shape)+[1]
                probs, v, logits, exp_val, sum_val, valids2= self.nnet.predict(
                    np.concatenate([np.reshape(raw_board,shape),
                                    np.reshape(mtx, shape),
                                    heuristic_components], axis=2),curPlayer)

                eps = self.args.heur_val_eps
                board = np.copy(raw_board)
                v_heur = self.get_heuristic_end(board, curPlayer, action)
                v = np.clip( (1-eps)*v[0] + eps*v_heur , -1,1)

            #v = v[0]
            # === Add Dirichlet noise to pi: ===
            eps = self.args.epsilon
            noise = np.random.dirichlet(self.alphas)
            probs = eps*noise+(1.0-eps)*probs

            # === Mask invalids ===
            valids = self.game.getValidMoves(raw_board, curPlayer)
            self.Ps[s] = probs*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # NB! All valid moves may be masked if either your NNet architecture
                # is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay
                # attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] +\
                    self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?
                    #u = self.Ps[s][a]
                if u > cur_best:
                    cur_best = u
                    best_act = a
                    
        a = best_act
        next_s, next_player = self.game.getNextState(raw_board, curPlayer, a)

        v = self.search(next_s, next_player, a)
        #Heuristic: v = self.Ps[s][a], cpuct = 0.0, u = self.Ps[s][a]

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v

    def get_heuristic_end(self, next_s, next_player, action):
        while 1:
            mtx = self.heuristic.get_field_stregth_mtx(next_s, 1)
            valids = self.game.getValidMoves(next_s, next_player)
            probs = mtx.flatten()
            eps = 0.1
            noise = np.random.dirichlet(self.alphas)*valids
            probs = eps*noise+(1.0-eps)*probs
 

            a = np.argmax(probs)
            next_s, next_player = self.game.getNextState(next_s, next_player, a)

            end = self.game.getGameEnded(next_s, next_player, a)
            if probs[a]==0.0:
                end=-1
            
            if end != 0:
                return end
                
        print("You shouldn't be here")
        return 1
    
    def evalSituation(self, raw_board, curPlayer, action):
        # TODO:
        #     - speedup with Es, Ps, Vs
        #     - disadvantage???
        next_s, next_player = raw_board, curPlayer
        a = action
        i=0
        probs0 = None
        while i < self.args.evaluationDepth:
            mtx = self.heuristic.get_field_stregth_mtx(next_s, 1)
            heuristic_components = self.heuristic.get_x_line_mtx(next_s, 1)
            shape = list(np.shape(next_s))+[1]
            valids = self.game.getValidMoves(next_s, next_player)

            if np.sum(heuristic_components[...,0])>0:
                # Move the only choice:
                a = np.argmax(heuristic_components[...,0].flatten())
                probs = np.zeros((self.game.getActionSize(),))
                probs[a] = 1
                #print("Obligatory step:")
                #display(next_s, end = True)
            else:
                probs, v, logits, exp_val, sum_val, valids2 = self.nnet.predict(
                    np.concatenate([np.reshape(next_s,shape),
                                np.reshape(mtx, shape),
                                heuristic_components], axis=2),next_player)
                i+=1

            if probs0 is None:
                probs0 = probs
            a = np.argmax(probs*valids)
            next_s, next_player = self.game.getNextState(next_s, next_player, a)

            # === Check, whether the game is over: ===
            reward = self.game.getGameEnded(next_s, next_player, a)
            if reward != 0:
                return probs0, reward
        return probs0, v
