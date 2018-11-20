# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 17:08
# @Author  : MengnanChen
# @FileName: MLP.py
# @Software: PyCharm

import tensorflow as tf

from models import hparams


class MLP:
    '''
    4 layer mlp for extracted feature of open-smile
    '''

    def __init__(self, mode, drop_rate=0.1):
        self.training = (mode == 'train')
        self.drop_rate = drop_rate
        self.layer_size = hparams.mlp_layer_size
        self.layer_units = hparams.mlp_layer_units

    def build_model(self, element):
        inputs, emo_labels = element
        assert len(self.layer_units) == self.layer_size
        if self.training:
            assert self.drop_rate is not None

        x = inputs
        self.input_y = emo_labels

        with tf.variable_scope('mlp'):
            for i in range(self.layer_size):
                x = tf.layers.dense(x, units=self.layer_units[i], activation=tf.nn.relu)
                if self.training:
                    x = tf.layers.dropout(x, rate=self.drop_rate)

        with tf.variable_scope('logits'):
            self.logits = tf.layers.dense(x, units=hparams.n_classes, activation=None)

    def add_loss(self):
        with tf.variable_scope('loss'):
            all_vars = tf.trainable_variables()
            self.regularization = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
                                            if not ('bias' in v.name or 'Bias' in v.name)]) * hparams.reg_weight

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                           logits=self.logits)
            self.loss = self.regularization + tf.reduce_mean(cross_entropy)

    def add_metric(self):
        with tf.variable_scope('metrics'):
            pred = tf.nn.softmax(logits=self.logits)
            self.pred_ids = tf.argmax(pred, axis=-1, output_type=tf.int32)
            # self.accuracy, _ = tf.metrics.accuracy(labels=self.input_y, predictions=self.pred_ids)
            correct_prediction=tf.equal(self.pred_ids,self.input_y)
            self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    def add_optimizer(self, global_step):
        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=hparams.initial_learning_rate,
                global_step=global_step,
                decay_rate=hparams.decay_rate,
                decay_steps=hparams.decay_steps
            )
            optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                               beta1=hparams.adam_beta1,
                                               beta2=hparams.adam_beta2,
                                               epsilon=hparams.adam_epsilon)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            if hparams.clip_gradients:
                clip_gradient, _ = tf.clip_by_global_norm(gradients, 1.)
            else:
                clip_gradient = gradients

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clip_gradient, variables),
                                                          global_step=global_step)
