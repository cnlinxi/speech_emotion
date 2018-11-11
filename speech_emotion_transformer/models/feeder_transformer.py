# -*- coding: utf-8 -*-
# @Time    : 2018/11/10 17:50
# @Author  : MengnanChen
# @FileName: feeder_transformer.py
# @Software: PyCharm

import threading
import time
from six.moves import cPickle as pickle

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from speech_emotion_transformer.models import hparams

_batches_per_group = 8


class Feeder:
    def __init__(self, coordinator, metadata_filepath):
        super(Feeder, self).__init__()
        self._coord = coordinator
        self._train_offset = 0
        self._test_offset = 0
        self._labels = hparams.classes

        # load metadata
        with open(metadata_filepath, 'rb') as fin:
            lines = fin.readlines()[1:]  # skip header
            self._metadata = [line.decode('utf-8').strip().split(',') for line in lines]
            print('metadata loaded...')
            # emo_labels=[]
            # pik_paths=[]
            # for line in lines:
            #     meta=line.split(',')
            #     emo_labels.append(meta[1])
            #     pik_paths.append(meta[3])
            # emo_labels=np.asarray(emo_labels)
            # pik_paths=np.asarray(pik_paths)

        test_size = (hparams.test_batch_size if hparams.test_batch_size is not None
                     else int(hparams.batch_size * hparams.test_batch_size_percent))
        indices = np.arange(len(self._metadata))
        train_indices, test_indices = train_test_split(indices,
                                                       test_size=test_size,
                                                       random_state=2018)
        len_test_indices = self._round_down(len(test_indices), hparams.batch_size)
        extra_test = test_indices = test_indices[len_test_indices:]
        test_indices = test_indices[:len_test_indices]
        train_indices = np.concatenate([train_indices, extra_test])
        self._train_meta = list(np.array(self._metadata)[train_indices])
        self._test_meta = list(np.array(self._metadata)[test_indices])

        self.test_steps = len(self._test_meta) // hparams.batch_size

        self._pad = 0.

        with tf.device('/cpu:0'):
            self._placeholders = [
                # [batch_size,n_frame,frame_size]
                tf.placeholder(tf.float32, shape=(None, None, hparams.frame_size), name='inputs'),
                tf.placeholder(tf.int32, shape=(None,), name='input_lengths'),  # [batch_size]
                tf.placeholder(tf.int32, shape=(None, hparams.n_classes), name='labels')  # [batch_size,n_classes]
            ]

            # train
            queue = tf.FIFOQueue(
                capacity=8,
                dtypes=[tf.float32,tf.int32,tf.int32],
                name='input_queue'
            )
            self._enqueue_op = queue.enqueue(vals=self._placeholders)
            self.inputs, self.input_lengths, self.labels = queue.dequeue()

            self.inputs.set_shape(self._placeholders[0].shape)
            self.input_lengths.set_shape(self._placeholders[1].shape)
            self.labels.set_shape(self._placeholders[2].shape)

            # eval
            eval_queue = tf.FIFOQueue(1, [tf.float32, tf.int32, tf.int32], name='eval_queue')

            self._eval_enqueue_op = queue.enqueue(self._placeholders)
            self.eval_inputs, self.eval_input_lengths, self.eval_labels = eval_queue.dequeue()

            self.eval_inputs.set_shape(self._placeholders[0].shape)
            self.eval_input_lengths.set_shape(self._placeholders[1].shape)
            self.eval_labels.set_shape(self._placeholders[2].shape)

    def start_threads(self, session):
        self._session = session
        thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
        thread.daemon = True  # thread will close when parent quits
        thread.start()

        thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
        thread.daemon = True
        thread.start()

    def _enqueue_next_train_group(self):
        while not self._coord.should_stop():
            start_time = time.time()
            n = hparams.batch_size
            examples = [self._get_next_example() for _ in range(n * _batches_per_group)]

            batches = [examples[i:i + n] for i in range(0, len(examples), n)]
            np.random.shuffle(batches)
            print('\ngenerate {} train batches of size {} in {:.3f} sec'.format(len(batches), n,
                                                                                time.time() - start_time))
            for batch in batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))
                self._session.run(self._enqueue_op, feed_dict=feed_dict)

    def _enqueue_next_test_group(self):
        # create test batches once and evaluate on them for all test steps
        test_batches = self.make_test_batches()
        while not self._coord.should_stop():
            for batch in test_batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))
                self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

    def _get_next_example(self):
        '''
        get one sample
        '''
        if self._train_offset >= len(self._train_meta):
            self._train_offset = 0
            np.random.shuffle(self._train_meta)
        meta = self._train_meta[self._train_offset]
        self._train_offset += 1

        emo_label = meta[1]
        audio_path = meta[3]
        with open(audio_path, 'rb') as fin:
            audio_data, sr = pickle.load(fin)
        emo_vec = [1 if emo_label == self._labels[i] else 0 for i in range(len(self._labels))]
        return (audio_data, emo_vec)

    def make_test_batches(self):
        start_time = time.time()

        n = hparams.batch_size
        examples = [self._get_test_example() for i in range(len(self._test_meta))]

        batches = [examples[i:i + n] for i in range(0, len(examples), n)]
        np.random.shuffle(batches)
        print('\ngenerate {} test batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start_time))
        return batches

    def _get_test_example(self):
        meta = self._test_meta[self._test_offset]
        self._test_offset += 1

        emo_label = meta[1]
        audio_path = meta[3]
        with open(audio_path, 'rb') as fin:
            audio_data, sr = pickle.load(fin)
        emo_vec = [1 if emo_label == self._labels[i] else 0 for i in range(len(self._labels))]
        return (audio_data, emo_vec)

    def _prepare_batch(self, batch):
        # batch: (audio_datas,emo_vecs)
        np.random.shuffle(batch)

        inputs = self._prepare_inputs([x[0] for x in batch])  # audio_data
        input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
        emo_labels = np.asarray([x[1] for x in batch])  # emo_labels
        return (inputs, input_lengths, emo_labels)

    def _prepare_inputs(self, inputs):
        integral_inputs = []
        for i in inputs:
            # pad input to integral frame, round up(x): multiple: frame_size
            integral_inputs.append(self._pad_input_to_integral_frame(i))
        integral_inputs = [np.reshape(x, (-1, hparams.frame_size)) for x in
                           integral_inputs]  # [batch_size,n_frame,frame_size]
        integral_inputs=integral_inputs[:,:hparams.max_n_frame,:]  # reduce too long audio
        max_len = max([len(x) for x in integral_inputs])
        # pad input to max_frame_length
        return np.stack([self._pad_input_to_max_frame_len(x, max_len) for x in integral_inputs])

    def _pad_input_to_integral_frame(self, x):
        length = self._round_up(x.shape[0], hparams.frame_size)
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

    def _pad_input_to_max_frame_len(self, x, length):
        return np.pad(x, [(0, length - x.shape[0]), (0, 0)], mode='constant', constant_values=self._pad)

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _round_down(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder
