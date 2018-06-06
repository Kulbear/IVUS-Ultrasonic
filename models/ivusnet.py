import tensorflow as tf

from .ops import *
from base.base_model import BaseModel
from tensorflow import layers as L

import tensorlayer as tl


def _net(input_tensor, is_training=True, config={}):
    he_init = tf.keras.initializers.he_normal()

    if config['data_format'] == 'NCHW':
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'

    if config['activation'] == 'prelu':
        activation = tf.nn.leaky_relu
    else:
        activation = tf.nn.relu

    if config['pooling'] == 'avg':
        pooling = tf.nn.avg_pool
    else:
        pooling = tf.nn.max_pool

    # encoder
    net = input_tensor
    enc_lyr_names = ['Enc_1', 'Enc_2', 'Enc_3', 'Enc_4', 'Enc_5', 'Enc_6']
    enc_lyr_depth = [16, 32, 64, 128, 256, 512]
    enc_lyrs = {}
    assert len(enc_lyr_names) == len(enc_lyr_depth)
    for idx in range(len(enc_lyr_names)):
        lyr_name = enc_lyr_names[idx]
        # print(lyr_name)
        with tf.variable_scope(lyr_name):
            # downsampling path
            if lyr_name != 'Enc_1':
                net = downsampling_branch(net, enc_lyr_depth[idx], pooling,
                                          activation, is_training, lyr_name)

            # refining branch
            ref = refining_branch(
                net,
                enc_lyr_depth[idx],
                pooling,
                activation,
                is_training,
                lyr_name,
                init=he_init)

            # main branch
            net = main_branch(
                net,
                ref,
                enc_lyr_depth[idx],
                pooling,
                activation,
                is_training,
                lyr_name,
                init=he_init)
            enc_lyrs[lyr_name] = net
            print(net)

    # decoder
    dec_lyr_names = ['Dec_5', 'Dec_4', 'Dec_3', 'Dec_2', 'Dec_1']
    dec_lyr_depth = [256, 128, 64, 32, 16]
    assert len(dec_lyr_names) == len(dec_lyr_depth)
    for idx in range(len(dec_lyr_names)):
        lyr_name = dec_lyr_names[idx]
        # print(lyr_name)
        with tf.variable_scope(lyr_name):
            net = upsampling_branch(
                net,
                dec_lyr_depth[idx],
                activation,
                is_training,
                lyr_name,
                init=he_init)

            # refining branch
            ref = refining_branch(
                net,
                dec_lyr_depth[idx],
                pooling,
                activation,
                is_training,
                lyr_name,
                init=he_init)
            net = tf.concat(
                axis=1, values=[enc_lyrs['Enc_{}'.format(lyr_name[-1])], net])
            # main branch
            net = main_branch(
                net,
                ref,
                dec_lyr_depth[idx],
                pooling,
                activation,
                is_training,
                lyr_name,
                init=he_init)
            print(net)

    # restore to original size
    net = restoring_branch(
        net, 16, activation, is_training, init=he_init, name='Res1')
    # output layer
    net = L.conv2d(
        net,
        1, [5, 5],
        strides=1,
        padding='SAME',
        kernel_initializer=he_init,
        data_format='channels_first',
        name='Output')
    net = tf.sigmoid(net, name='Output_Sigmoid')
    print(net)
    net = tf.reshape(net, (-1, 512, 512))

    return net


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.build_model()
        self.init_saver()

    def predict(self, sess, features):
        feed_dict = {self.x: features, self.is_training: False}
        # prediction = tf.argmax(self.logits, axis=-1)
        pred = sess.run(self.logits, feed_dict=feed_dict)
        return pred

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, ())
        state_size = self.config['state_size']
        data_format = self.config['data_format']
        if data_format == 'NCHW':
            self.x = tf.placeholder(
                tf.float32, (None, 1, *state_size[:-1]), name='x')
        else:
            self.x = tf.placeholder(
                tf.float32, (None, *state_size[:-1], 1), name='x')

        self.y = tf.placeholder(tf.float32, (None, 512, 512), name='y')

        self.logits = _net(self.x, config=self.config)

        with tf.name_scope('loss'):
            # print(self.logits, self.y)
            self.cross_entropy = 1 - tl.cost.dice_coe(
                self.logits, self.y, axis=[1, 2])
            self.train_step = tf.train.AdamOptimizer(
                self.config.learning_rate).minimize(
                    self.cross_entropy, global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
