# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 17:08
# @Author  : MengnanChen
# @FileName: MLP.py
# @Software: PyCharm

import tensorflow as tf

from speech_emotion_transformer.models import hparams


class MLP:
    '''
    4 layer mlp for extracted feature of open-smile
    '''
    def __init__(self, mode,drop_rate=0.1):
        self.mode = mode
        self.drop_rate = drop_rate
        self.input_x = tf.placeholder(tf.float32, shape=[None, hparams.n_features])  # [batch_size,n_features]
        self.input_y = tf.placeholder(tf.int32, shape=[None])  # [batch_size]
        self.layer_size = 4
        self.layer_units = (512, 1024, 512, 256)

        self.n_classes=4

    def _build_model(self):
        assert len(self.layer_units) == self.layer_size
        if self.mode == 'train':
            assert self.drop_rate is not None

        x = self.input_x

        with tf.variable_scope('mlp'):
            for i in range(self.layer_size):
                x = tf.layers.dense(x, units=self.layer_units[i], activation=tf.nn.relu)
                if self.mode == 'train':
                    x = tf.layers.dropout(x, rate=self.drop_rate)

        with tf.variable_scope('logits'):
            self.logits=tf.layers.dense(x,units=self.n_classes,activation=None)
