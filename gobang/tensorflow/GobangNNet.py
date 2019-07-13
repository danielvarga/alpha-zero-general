import sys
sys.path.append('..')
from utils import *

import tensorflow as tf

class GobangNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Renaming functions 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.keras.layers.BatchNormalization
        Dropout = tf.keras.layers.Dropout
        Dense = tf.keras.layers.Dense

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y])    # s: batch_size x board_x x board_y
            self.curPlayer = tf.placeholder(tf.float32, shape=[None,1])
            self.dropout = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool, name="is_training")


            x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, 1])                    # batch_size  x board_x x board_y x 1
            white_image = x_image * (x_image + 1) / 2
            black_image = x_image * (x_image -1) / 2
            
            players = tf.reshape(self.curPlayer, [-1, 1, 1, 1])
            player_channel = players * tf.ones_like(x_image)
            
            x_image = tf.concat([white_image, black_image, player_channel], axis=3)
            h_conv1 = Relu(BatchNormalization(axis=3)(self.conv2d(x_image, args.num_channels, 'same'), training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
            h_conv2 = Relu(BatchNormalization(axis=3)(self.conv2d(h_conv1, args.num_channels, 'same'), training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
            h_conv3 = Relu(BatchNormalization(axis=3)(self.conv2d(h_conv2, args.num_channels, 'same', strides=(2,1)), training=self.isTraining))    # batch_size  x (board_x/2) x (board_y) x num_channels
            h_conv4 = Relu(BatchNormalization(axis=3)(self.conv2d(h_conv3, args.num_channels, 'same', strides=(2,2)), training=self.isTraining))    # batch_size  x (board_x/4) x (board_y/2) x num_channels
            h_conv4_flat = tf.reshape(h_conv4, [-1, args.num_channels*(self.board_x//4)*(self.board_y//2)])

            # alternative information channel
            # x_flat = tf.reshape(self.input_boards, [-1, self.board_x * self.board_y])
            # x_flat = tf.concat([x_flat, self.curPlayer], axis=1)
            # print("xflat_shape: ", x_flat.shape)
            # h_fc1 = Dropout(self.dropout)(Relu(BatchNormalization(axis=1)(Dense(512)(x_flat), training=self.isTraining))) # batch_size x 1024
            # h_fc2 = Dropout(self.dropout)(Relu(BatchNormalization(axis=1)(Dense(256)(h_fc1), training=self.isTraining)))         # batch_size x 512
            # h = tf.keras.layers.concatenate([h_conv4_flat, h_fc2], axis=-1)
            h = h_conv4_flat
            
            s_fc1 = Dropout(self.dropout)(Relu(BatchNormalization(axis=1)(Dense(1024)(h), training=self.isTraining))) # batch_size x 1024
            s_fc2 = Dropout(self.dropout)(Relu(BatchNormalization(axis=1)(Dense(512)(s_fc1), training=self.isTraining)))         # batch_size x 512
            self.pi = Dense(self.action_size)(s_fc2)                                                        # batch_size x self.action_size
            self.prob = tf.nn.softmax(self.pi)
            self.v = Tanh(Dense(1)(s_fc2))                                                               # batch_size x 1

            self.calculate_loss()

    def conv2d(self, x, out_channels, padding, strides=(1,1)):
        return tf.keras.layers.Conv2D(out_channels, kernel_size=[4,4], strides=strides, padding=padding)(x)

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)




