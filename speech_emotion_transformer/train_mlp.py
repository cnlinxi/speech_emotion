# -*- coding: utf-8 -*-
# @Time    : 2018/11/18 16:22
# @Author  : MengnanChen
# @FileName: train_mlp.py
# @Software: PyCharm
import os
import time
import traceback
from datetime import datetime

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from models import hparams
from models import create_model
from models.feeder_mlp import Feeder


class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []


def add_train_stats(model):
    with tf.variable_scope('stats') as scope:
        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('learning_rate', model.learning_rate)

        return tf.summary.merge_all()


def add_eval_stats(summary_writer, step, loss, acc):
    values = [
        tf.Summary.Value(tag='eval_model/eval_stats/eval_loss', simple_value=loss),
        tf.Summary.Value(tag='eval_model/eval_stats/eval_acc', simple_value=acc)
    ]
    test_summary = tf.Summary(value=values)
    summary_writer.add_summary(test_summary, step)


def model_train_mode(feeder, global_step):
    with tf.variable_scope('ser_mlp', reuse=tf.AUTO_REUSE):
        model = create_model(mode='train', model_type='mlp')
        model.build_model(feeder.next_element)
        model.add_loss()
        model.add_metric()
        model.add_optimizer(global_step)
        stats = add_train_stats(model)
    return model, stats


def model_test_mode(feeder):
    with tf.variable_scope('ser_mlp_eval', reuse=tf.AUTO_REUSE):
        model = create_model(mode='test', model_type='mlp')
        model.build_model(feeder.eval_next_element)
        model.add_loss()
        model.add_metric()
    return model


def train(log_dir):
    save_dir = os.path.join(log_dir, 'ckpt')
    tensorboard_dir = os.path.join(log_dir, 'events')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'ser_mlp.ckpt')

    print('checkpoint path: {}'.format(checkpoint_path))
    tf.set_random_seed(seed=2018)

    print('train model set to max steps: {}'.format(hparams.train_steps))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        try:
            feeder = Feeder(data_path=hparams.emo_opensmile_data_path, sess=sess)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            model, stats = model_train_mode(feeder, global_step)
            eval_model = model_test_mode(feeder)

            step = 0
            time_window = ValueWindow(100)
            loss_window = ValueWindow(100)
            saver = tf.train.Saver(max_to_keep=5)

            summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
            sess.run((tf.global_variables_initializer(),tf.local_variables_initializer()))
            if hparams.restore:
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                    if checkpoint_state and checkpoint_state.model_checkpoint_path:
                        print('loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
                        saver.restore(sess, checkpoint_state.model_checkpoint_path)
                    else:
                        print('no model at {}'.format(save_dir))
                        saver.save(sess, checkpoint_path, global_step=global_step)
                except tf.errors.OutOfRangeError as e:
                    print('cannot restore checkpoint :{}'.format(e))
            else:
                print('start new train!')
                saver.save(sess, checkpoint_path, global_step=global_step)

            while step < hparams.train_steps:
                start_time = time.time()
                step, loss, opt = sess.run([global_step, model.loss, model.optimize])
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
                    step, time_window.average, loss, loss_window.average)
                # print(message, end='\r')

                # if loss > 1e10 or np.isnan(loss):
                    # print('loss exploded to {:.5f} at step {}'.format(loss, step))
                    # raise Exception('loss exploded')
                if step % hparams.summary_interval == 0:
                    # print('\nwrite summary at step {}'.format(step))
                    summary_writer.add_summary(sess.run(stats), step)
                if step % hparams.eval_interval == 0:
                    print('run evaluation at step {}'.format(step))
                    eval_losses = []
                    eval_accuracys = []

                    for i in tqdm(range(feeder.eval_num_batch)):
                        ground_truth_ids, pred_ids, loss, accuracy = sess.run(
                            [eval_model.input_y, eval_model.pred_ids, eval_model.loss, eval_model.accuracy])

                        print('grou ids: {}'.format(ground_truth_ids))
                        print('pred ids: {}'.format(pred_ids))
                        print('batch-{} acc: {}'.format(i,accuracy))

                        eval_losses.append(loss)
                        eval_accuracys.append(accuracy)

                    eval_loss = sum(eval_losses) / len(eval_losses)
                    eval_accuracy = sum(eval_accuracys) / len(eval_accuracys)

                    add_eval_stats(summary_writer, step, eval_loss, eval_accuracy)
                    print('\nacc: {}'.format(eval_accuracy))

                if step % hparams.checkpoint_interval == 0 or step == hparams.train_steps \
                        or step == 300:
                    saver.save(sess, checkpoint_path, global_step=global_step)

            print('train complete after {} global steps'.format(hparams.train_steps))
            return save_dir
        except Exception as e:
            print('exiting due to exception: {}'.format(e))
            traceback.print_exc()


def main(base_dir, run_name):
    log_dir = os.path.join(base_dir, 'logs_{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)

    print('#' * 32)
    print('\nmlp train\n')
    print('#' * 32)
    checkpoint_path = train(log_dir)
    tf.reset_default_graph()
    time.sleep(0.5)
    if checkpoint_path is None:
        raise Exception('Exiting!')


if __name__ == '__main__':
    base_dir = 'data/model'
    os.makedirs(base_dir, exist_ok=True)
    run_name = 'ser_{}'.format(datetime.now().strftime('%Y_%m_%d_%H_%M'))
    main(base_dir, run_name)
