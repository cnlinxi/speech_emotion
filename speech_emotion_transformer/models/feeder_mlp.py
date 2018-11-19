# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 20:29
# @Author  : MengnanChen
# @FileName: feeder_mlp.py
# @Software: PyCharm

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from models import hparams


class Feeder(object):
    def __init__(self, data_path, sess):
        super(Feeder, self).__init__()
        self._data_path = data_path

        data = pd.read_csv(self._data_path)
        label_name = 'emolabel'
        data[label_name] = LabelEncoder().fit_transform(data[label_name])
        feature_column_names = [x for x in data.columns if x != 'emolabel']
        test_size = hparams.test_batch_size if hparams.test_batch_size is not None \
            else hparams.test_batch_size_percent
        X_train, X_test, y_train, y_test = train_test_split(data[feature_column_names].values,
                                                            data['emolabel'].values,
                                                            test_size=test_size)

        self.num_batch = len(X_train) // hparams.batch_size
        self.eval_num_batch = len(X_test) // hparams.batch_size

        self._placeholder = (
            tf.placeholder(tf.float32, shape=(None, hparams.n_features), name='input_x'),
            tf.placeholder(tf.int32, shape=(None,), name='input_y')
        )

        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices(self._placeholder)
            dataset = dataset.batch(hparams.batch_size).shuffle(buffer_size=1000).repeat()
            iterator = dataset.make_initializable_iterator()
            sess.run(iterator.initializer, feed_dict=dict(zip(self._placeholder, (X_train, y_train))))
            self.next_element = iterator.get_next()

            eval_dataset = tf.data.Dataset.from_tensor_slices(self._placeholder)
            eval_dataset = eval_dataset.batch(hparams.batch_size).repeat()
            eval_iterator = eval_dataset.make_initializable_iterator()
            sess.run(eval_iterator.initializer, feed_dict=dict(zip(self._placeholder, (X_test, y_test))))
            self.eval_next_element = eval_iterator.get_next()
        # X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        # y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
        #
        # input_queues = tf.train.slice_input_producer([X_train, y_train])
        # self.input_x, self.input_y = tf.train.shuffle_batch(input_queues,
        #                                                     num_threads=2,
        #                                                     batch_size=hparams.batch_size,
        #                                                     capacity=hparams.batch_size * 64,
        #                                                     min_after_dequeue=hparams.batch_size * 32,
        #                                                     allow_smaller_final_batch=False)

        # X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        # y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
        # eval_input_queues = tf.train.slice_input_producer([X_test, y_test])
        # self.eval_input_x, self.eval_input_y = tf.train.shuffle_batch(eval_input_queues,
        #                                                               num_threads=2,
        #                                                               batch_size=hparams.batch_size,
        #                                                               capacity=hparams.batch_size * 64,
        #                                                               min_after_dequeue=hparams.batch_size * 32,
        #                                                               allow_smaller_final_batch=False)
