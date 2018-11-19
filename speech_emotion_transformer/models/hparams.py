# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 17:22
# @Author  : MengnanChen
# @FileName: hparams.py
# @Software: PyCharm

classes = ['exc', 'ang', 'sad', 'neu', 'fru']
n_classes = 5  # number of classes， len(classes)
metadata_filepath = 'data/iemocap_raw_output_new.csv'
raw_audio_root_dir = 'data/raw_audio_data_new'
eval_data_dir='data/eval_data'
emo_opensmile_data_path='data/iemocap_output_emo_large_new.csv'
lgb_model_save_dir='../data/model'


n_features = 6553  # number of open smile extracted feature

frame_duration = 50  # ms
sr = 16000  # sample rate
frame_size = 80  # frame_duration*sr, number of sample points per frame

max_n_frame = 640  # sr*(max_time=3s)/frame_size, the best is integral multiple of frame_size
min_n_frame = 240  # reduce too short audio in preprocess
trim_top_db = 23
trim_fft_size = 512
trim_hop_size = 128

transformer_drop_rate = 0.1
transformer_hidden_units = frame_size
transformer_num_blocks = 4
transformer_num_heads = 4  # assert num_units%num_heads==0 in transformer_modules.py

blstm_cell_num=256
final_dense_units=(512,256)

mlp_dropout_rate=0.1
mlp_layer_units=(512, 1024, 512, 256)
mlp_layer_size=len(mlp_layer_units)

batch_size = 2
test_batch_size = 4  # test size
test_batch_size_percent = 0.005  # test on 0.5% of all dataset, take effect if test_batch_size is None

reg_weight = 1e-3  # l2 regularization weight
initial_learning_rate = 1e-3  # initial learning rate
decay_rate = 0.96
decay_steps = 10000
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-6
clip_gradients = True

train_steps = 10000
restore = True
summary_interval = 250
eval_interval = 100
checkpoint_interval = 3000
