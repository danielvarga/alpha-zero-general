import sys
sys.path.append('..')
from utils import *

import tensorflow as tf


# Simple Network
# Convolutional Network
class GobangNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default():
            channel_num = 9
            self.input_boards = tf.placeholder(tf.float32,
                                               shape=[None, self.board_x, self.board_y, channel_num],
                                               name = "input_boards")

            self.curPlayer = tf.placeholder(tf.float32, shape=[None,1], name = "curPlayer")
            self.dropout = tf.placeholder(tf.float32, name="dropout")
            self.isTraining = tf.placeholder(tf.bool, name="isTraining")

            self.build_conv_model(channel_num)
            #self.build_dense_model(channel_num)
            
            self.calculate_loss()

            # Collect logging info:
            #self.logits = self.pi
            #self.exp_val = self.pi
            #self.sum_val = self.pi
            #self.valids = self.pi
            self.input=self.prob
            self.output=self.prob

    def conv2d(self, x, out_channels, padding, strides=(1,1)):
        return tf.keras.layers.Conv2D(out_channels, kernel_size=[4,4],
                                      strides=strides, padding=padding)(x)

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        #self.loss_pi =  tf.losses.mean_squared_error(self.target_pis, self.prob)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.args.pi_weight*self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
            gvs = optimizer.compute_gradients(self.total_loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_step = optimizer.apply_gradients(capped_gvs)

    def valid_softmax(self, logits, valids):
        # Need to be logged:
        self.logits = logits
        
        logits -= tf.reduce_min(logits, axis = -1, keepdims = True)
        m = tf.math.reduce_max(logits*valids, axis = -1, keepdims = True)
    
        # Find max of valid logits:
        logits -= tf.reduce_min(logits, axis = -1, keepdims = True)
        m = tf.math.reduce_max(logits*valids, axis = -1, keepdims = True)
        # Clip invalid logits to m
        logits = tf.math.minimum(logits, m)-m
        exp_val = valids*tf.exp(logits)
        sum_val = tf.math.reduce_sum(exp_val, axis = -1, keepdims=True)

        self.exp_val = exp_val
        self.sum_val = sum_val
        
        return exp_val/sum_val
    
    def build_dense_model(self, channel_num):
        # Renaming functions 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.keras.layers.BatchNormalization
        Dropout = tf.keras.layers.Dropout
        Dense = tf.keras.layers.Dense

        x_flat = tf.reshape(self.input_boards, [-1, self.board_x * self.board_y*channel_num])
        x_flat = tf.concat([x_flat, self.curPlayer], axis=1)

        dense_1 = Dropout(rate=self.dropout)(Relu(Dense(512)(x_flat)))
        dense_2 = Dropout(rate=self.dropout)(Relu(Dense(256)(dense_1)))
        dense_3 = Dropout(rate=self.dropout)(Relu(Dense(128)(dense_2)))

        self.pi = Dense(self.action_size)(dense_3)                # batch_size x self.action_size
        self.prob = tf.nn.softmax(self.pi)
        self.v = Tanh(Dense(1)(dense_3))                            # batch_size x 1
        
    def build_conv_model(self, channel_num):
        args = self.args
        # Renaming functions 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.keras.layers.BatchNormalization
        Dropout = tf.keras.layers.Dropout
        Dense = tf.keras.layers.Dense

        x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, channel_num])

        # Preprocess and slice simple-board, and heuristic-board
        board = tf.slice(x_image, [0,0,0,0], [-1, self.board_x, self.board_y, 1])
        heur_channels = tf.slice(x_image,
                                    [0, 0,0,2],
                                    [-1, self.board_x, self.board_y, channel_num-3])
        white_image = board * (board + 1) -1
        black_image = board * (board - 1) -1

        empty_fields = -tf.math.abs(board)+1
        flat_empty_fields = tf.reshape(empty_fields, [-1, self.board_x*self.board_y])
        paddings = tf.constant([[0, 0,], [0, 1]])
        self.valids = tf.pad(flat_empty_fields, paddings, "CONSTANT")
        
        players = tf.reshape(self.curPlayer, [-1, 1, 1, 1])
        player_channel = players * tf.ones_like(x_image)
        
        x_image = tf.concat([white_image, black_image, heur_channels, player_channel], axis=3)

        batchNorm=False
        flat_dim = self.args.num_channels*(self.board_x//4)*(self.board_y//2)
        if(batchNorm):
            h_conv1 = Relu(BatchNormalization(axis=3)(self.conv2d(x_image, args.num_channels,
                    'same'), training=self.isTraining))     # Batch * X * Y * Channel
            h_conv2 = Relu(BatchNormalization(axis=3)(self.conv2d(h_conv1, args.num_channels,
                    'same'), training=self.isTraining))     # Batch * X * Y * Channel
            h_conv3 = Relu(BatchNormalization(axis=3)(self.conv2d(h_conv2, args.num_channels,
                    'same', strides=(2,1)), training=self.isTraining)) # Batch*X/2*Y*Channel
            h_conv4 = Relu(BatchNormalization(axis=3)(self.conv2d(h_conv3, args.num_channels,
                    'same', strides=(2,2)), training=self.isTraining))
            # Batch*X/4*Y/2*Channel
            h_conv4_flat = tf.reshape(h_conv4, [-1, flat_dim])
        else:
            # Batch * X * Y * Channel
            h_conv1 = Relu(self.conv2d(x_image, args.num_channels, 'same'))
            h_conv2 = Relu(self.conv2d(h_conv1, args.num_channels, 'same'))
            h_conv3 = Relu(self.conv2d(h_conv2, args.num_channels, 'same', strides=(2,1)))
            h_conv4 = Relu(self.conv2d(h_conv3, args.num_channels, 'same', strides=(2,2)))
            # Batch * X/4 * Y/2 * Channel
            h_conv4_flat = tf.reshape(h_conv4, [-1, flat_dim])

        # alternative information channel
        x_flat = tf.reshape(self.input_boards, [-1, self.board_x * self.board_y*channel_num])
        x_flat = tf.concat([x_flat, self.curPlayer], axis=1)
        if(batchNorm):
            h_fc1 = Dropout(rate=self.dropout)(
                    Relu(
                    BatchNormalization(axis=1)(
                    Dense(512)(x_flat),training=self.isTraining)))
            h_fc2 = Dropout(rate=self.dropout)(
                    Relu(
                    BatchNormalization(axis=1)(
                    Dense(256)(h_fc1),training=self.isTraining)))
        else:
            h_fc1 = Dropout(rate=self.dropout)(Relu(Dense(512)(x_flat)))
            h_fc2 = Dropout(rate=self.dropout)(Relu(Dense(256)(h_fc1)))
        h = tf.keras.layers.concatenate([h_conv4_flat, h_fc2], axis=-1)
        #h = h_fc2

        if(batchNorm):
            s_fc1 = Dropout(rate=self.dropout)(Relu(BatchNormalization(axis=1)(Dense(1024)(h),
                            training=self.isTraining)))         # batch_size x 1024
            s_fc2 = Dropout(rate=self.dropout)(Relu(BatchNormalization(axis=1)(Dense(512)(s_fc1),
                            training=self.isTraining)))         # batch_size x 512
        else:
            s_fc1 = Dropout(rate=self.dropout)(Relu(Dense(1024)(h)))    # batch_size x 1024
            s_fc2 = Dropout(rate=self.dropout)(Relu(Dense(512)(s_fc1))) # batch_size x 512

        self.pi = Dense(self.action_size)(s_fc2)                # batch_size x self.action_size
        self.prob = tf.nn.softmax(self.pi, name = "prob")
        #self.prob = self.valid_softmax(self.pi, self.valids)
        self.v = Tanh(Dense(1)(s_fc2), name = "v")                          # batch_size x 1


