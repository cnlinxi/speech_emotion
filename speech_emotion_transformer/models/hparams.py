# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 17:22
# @Author  : MengnanChen
# @FileName: hparams.py
# @Software: PyCharm

classes = ['exc', 'ang', 'sad', 'neu', 'fru']
n_classes = 5  # number of classes， len(classes)

n_features = 384  # number of open smile extracted feature

frame_duration = 50  # ms
sr = 16000  # sample rate
frame_size = 80  # frame_duration*sr, number of sample points per frame

max_n_frame = 4000  # sr*(max_time=20s)/frame_size

transformer_drop_rate=0.1
transformer_hidden_units = frame_size
transformer_num_blocks = 4
transformer_num_heads = 6

batch_size = 32
test_batch_size = None
test_batch_size_percent = 0.2

reg_weight = 1e-3  # l2 regularization weight
initial_learning_rate = 1e-3  # initial learning rate
decay_rate = 0.96
decay_steps = 10000
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-6
clip_gradients = True

train_steps=100000
restore=True
summary_interval=250
eval_interval=5000
checkpoint_interval=3000
