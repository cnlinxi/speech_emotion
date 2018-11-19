# -*- coding: utf-8 -*-
# @Time    : 2018/11/10 15:05
# @Author  : MengnanChen
# @FileName: speech_transformer.py
# @Software: PyCharm

import tensorflow as tf

from models import hparams
from models.transformer_modules import *


class SpeechTransformer(object):
    def __init__(self, mode, drop_rate=0.1):
        self.mode = mode
        self.drop_rate = drop_rate

    def _build_model(self, inputs, input_lengths, emo_labels):
        '''
        build model
        :param inputs: [batch_size,n_frames,frame_size]
        :param input_lengths: [batch_size,]
        :param emo_labels: [batch_size,n_classes]
        '''
        if self.mode == 'train':
            assert self.drop_rate is not None
        training = (self.mode == 'train')
        self.input_x = inputs
        self.input_y = emo_labels
        max_n_frames = tf.cast(tf.ceil(tf.reduce_max(input_lengths) / hparams.frame_size), dtype=tf.int32)

        with tf.variable_scope('pos_enc'):
            pos_enc = positional_encoding(  # [batch_size,N,max_frame_len]
                inputs=self.input_x,
                num_units=hparams.transformer_hidden_units,
                zero_pad=False,
                scale=False
            )
            # pos_enc=pos_enc[:,:max_n_frames,:]
            pos_enc = tf.slice(pos_enc, begin=[0, 0, 0], size=[-1, max_n_frames, -1])
            self.input_x = self.input_x + pos_enc

            self.input_x = tf.layers.dropout(
                inputs=self.input_x,
                rate=self.drop_rate,
                training=training
            )

        for i in range(hparams.transformer_num_blocks):
            with tf.variable_scope('num_blocks_{}'.format(i)):
                self.input_x = multihead_attention(
                    queries=self.input_x,
                    keys=self.input_x,
                    num_units=hparams.transformer_hidden_units,
                    num_heads=hparams.transformer_num_heads,
                    dropout_rate=self.drop_rate,
                    is_training=training,
                    causality=False
                )
                self.input_x = feedforward(
                    inputs=self.input_x,
                    num_units=(4 * hparams.transformer_hidden_units, hparams.transformer_hidden_units)
                )
        # self.input_x = tf.layers.batch_normalization(self.input_x, training=training)  # [batch_size,N,T]

        # with tf.variable_scope('final_reshape'):
        # [batch_size,N]
        # shp = tf.shape(self.input_x)
        # flatten_shape = shp[1] * shp[2]
        # self.input_x = tf.reshape(self.input_x, [hparams.batch_size, flatten_shape], name='reshp')

        # x = tf.layers.conv1d(
        #     inputs=x,
        #     filters=1,
        #     strides=1,
        #     activation=None,
        #     padding='valid'
        # )  # [batch_size,N,1]
        # x = tf.squeeze(x)

        with tf.variable_scope('global_pool'):
            self.input_x=tf.reduce_max(tf.abs(self.input_x),reduction_indices=[1])
            self.input_x=tf.reshape(self.input_x,(-1,hparams.frame_size))
        # with tf.variable_scope('blstm'):
        #     cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hparams.blstm_cell_num)
        #     if training:
        #         cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw,
        #                                                 output_keep_prob=1. - hparams.transformer_drop_rate)
        #     cell_bw = tf.nn.rnn_cell.BasicLSTMCell(hparams.blstm_cell_num)
        #     if training:
        #         cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw,
        #                                                 output_keep_prob=1. - hparams.transformer_drop_rate)
        #
        #     outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
        #                                                              cell_bw=cell_bw,
        #                                                              inputs=self.input_x,
        #                                                              dtype=tf.float32,
        #                                                              time_major=False,
        #                                                              scope='blstm_rnn')
        #     output_states_fw, output_states_bw = output_states
        #     c_fw, h_fw = output_states_fw
        #     c_bw, h_bw = output_states_bw
        #     outputs = tf.concat((h_fw, h_bw), axis=-1)
        #     self.input_x = tf.reshape(outputs, (-1, 2 * hparams.blstm_cell_num))
            # outputs=tf.concat(output_states,axis=-1)
            # self.input_x=tf.reshape(outputs,(-1,4*hparams.blstm_cell_num))

        with tf.variable_scope('final_training_op'):
            # for unit in hparams.final_dense_units:
            #     self.input_x=tf.layers.dense(self.input_x,units=unit,activation=tf.nn.relu)
            #     self.input_x=tf.layers.dropout(self.input_x,rate=hparams.transformer_drop_rate)
            self.logits = tf.layers.dense(self.input_x, hparams.n_classes)

    def _add_loss(self):
        with tf.variable_scope('loss'):
            all_vars = tf.trainable_variables()
            self.regularization = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
                                            if not ('bias' in v.name or 'Bias' in v.name)]) * hparams.reg_weight
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,
                                                                            logits=self.logits)

            self.loss = self.regularization+tf.reduce_mean(self.cross_entropy)

    def _add_metric(self):
        with tf.variable_scope('metrics'):
            pred = tf.nn.softmax(logits=self.logits)
            self.pred_ids=tf.argmax(pred, axis=-1)
            self.ground_truth_ids=tf.argmax(self.input_y, axis=-1)
            correct_prediction = tf.equal(self.pred_ids,self.ground_truth_ids)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _add_optimizer(self, global_step):
        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=hparams.initial_learning_rate,
                global_step=global_step,  # add 1 if call optimizer once
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
                clipped_gradient, _ = tf.clip_by_global_norm(gradients, 1.)
            else:
                clipped_gradient = gradients

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradient, variables),
                                                          global_step=global_step)
