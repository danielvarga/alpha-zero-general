import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../../')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet

import tensorflow as tf
from .GobangNNet import GobangNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.0,
    'epochs': 10,
    'batch_size': 32,
    'num_channels': 128,
    'pi_weight':10.0,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game, displaybar=True):
        self.nnet = onnet(game, args)
        self.displaybar = displaybar
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.sess = tf.Session(graph=self.nnet.graph)
        self.saver = None
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.nnet.graph.get_collection('variables')))

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()
            bar = Bar('Training Net', max=int(len(examples)/args.batch_size))
            batch_idx = 0

            # self.sess.run(tf.local_variables_initializer())
            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, players, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                players = np.expand_dims(players, 1)
                #heuristic_boards = mcts.heuristic.get_field_stregth_mtx(board, 1)

                # predict and compute gradient and do SGD step
                input_dict = {self.nnet.input_boards: boards, self.nnet.target_pis: pis,
                              self.nnet.target_vs: vs, self.nnet.curPlayer: players,
                              self.nnet.dropout: args.dropout, self.nnet.isTraining: True,
                }

                # measure data loading time
                data_time.update(time.time() - end)

                # record loss
                self.sess.run(self.nnet.train_step, feed_dict=input_dict)
                pi_loss, v_loss, v, logits, exp_val, sum_val = self.sess.run(
                    [self.nnet.loss_pi, self.nnet.loss_v, self.nnet.v, self.nnet.logits,
                     self.nnet.exp_val, self.nnet.sum_val], feed_dict=input_dict)
                #print(input, output, pis)
                #print("target_v: {}".format(vs))
                #print(" model_v: {}".format(v))
                #print(v_loss)
                pi_losses.update(pi_loss, len(boards))
                v_losses.update(v_loss, len(boards))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                if self.displaybar:
                    # plot progress
                    bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                        batch=batch_idx,
                        size=int(len(examples)/args.batch_size),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        lpi=pi_losses.avg,
                        lv=v_losses.avg,
                    )
                    bar.next()
            bar.finish()

    def predict(self, board, curPlayer, multi = False):
        """
        board: np array with board
        """
        # timing
        # start = time.time()
       
        # preparing input
        if not multi:
            board = board[np.newaxis, :, :]
            curPlayer = np.array([[curPlayer]])
        else:
            curPlayer = curPlayer[:, np.newaxis]
        # run
        prob, v, logits, exp_val, sum_val, valids= self.sess.run(
            [self.nnet.prob, self.nnet.v, self.nnet.logits, self.nnet.exp_val, self.nnet.sum_val, self.nnet.valids],
            feed_dict={self.nnet.input_boards: board,
                       self.nnet.curPlayer:curPlayer,
                       self.nnet.dropout: 0.0, self.nnet.isTraining: False})

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        if multi:
            return prob
        else:
            return prob[0], v[0], logits[0], exp_val[0], sum_val[0], valids[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        if self.saver == None:            
            self.saver = tf.train.Saver(self.nnet.graph.get_collection('variables'))
        with self.nnet.graph.as_default():
            self.saver.save(self.sess, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath+'.meta'):
            raise("No model in path {}".format(filepath))
        with self.nnet.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, filepath)
