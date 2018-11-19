# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 18:00
# @Author  : MengnanChen
# @FileName: data_mod_utils.py
# @Software: PyCharm

import os
import sys
sys.path.append('../')
import math
from six.moves import cPickle as pickle

import numpy as np
import pandas as pd
from models import hparams
import librosa


def mod_meta(input_path, output_path):
    with open(input_path, 'rb') as fin:
        lines = fin.readlines()
        lines = [x.decode('utf-8').strip() for x in lines]
        lines_t = []
        for line in lines:
            line = line.split(',')
            line = ','.join(line[:3])
            lines_t.append('{}\n'.format(line))

    with open(output_path, 'wb') as fout:
        fout.writelines([x.encode('utf-8') for x in lines_t])


def trim_silence(wav):
    return librosa.effects.trim(wav, top_db=hparams.trim_top_db, frame_length=hparams.trim_fft_size,
                                hop_length=hparams.trim_hop_size)[0]


def bucket_examples(metadata_path, metadata_output_path, audio_root_dir, audio_root_dir_new):
    assert audio_root_dir_new != audio_root_dir
    print('process meta...')
    mod_meta(metadata_path, metadata_output_path)
    print('process examples...')
    metadata = pd.read_csv(metadata_output_path)
    audio_ids = metadata['wavname'].values
    emo_labels = metadata['emolabel'].values
    gender_labels = metadata['genderlabel'].values
    example_paths = [os.path.join(audio_root_dir, '{}.wav.pik'.format(x)) for x in audio_ids]

    os.makedirs(audio_root_dir_new, exist_ok=True)
    max_frame_size = hparams.max_n_frame * hparams.frame_size
    min_frame_size = hparams.min_n_frame * hparams.frame_size
    audio_ids_new = []
    emo_labels_new = []
    gender_labels_new = []
    example_lengths = []

    average_y=[]
    for ex, id, emo, gender in zip(example_paths, audio_ids, emo_labels, gender_labels):
        with open(ex, 'rb') as fin:
            y, sr = pickle.load(fin)
            y = trim_silence(y)
            average_y.append(np.mean(y))
        # reduce too long audio
        for i in range(math.ceil(y.shape[0] / max_frame_size)):
            if y.shape[0] >= max_frame_size:
                # new sample
                y_t = y[:max_frame_size]
                audio_id_new = '{}_{}'.format(id, str(i))
                audio_ids_new.append(audio_id_new)
                emo_labels_new.append(emo)
                gender_labels_new.append(gender)

                example_file_name = '{}.wav.pik'.format(audio_id_new)
                with open(os.path.join(audio_root_dir_new, example_file_name), 'wb') as fin:
                    pickle.dump(y_t, fin)
                example_lengths.append(len(y_t))

                y = y[max_frame_size:]
            elif y.shape[0] > min_frame_size:  # min_frame_size < y.shape[0] < max_frame_size
                audio_id_new = '{}_{}'.format(id, str(i))
                audio_ids_new.append(audio_id_new)
                emo_labels_new.append(emo)
                gender_labels_new.append(gender)

                example_file_name = '{}.wav.pik'.format(audio_id_new)
                with open(os.path.join(audio_root_dir_new, example_file_name), 'wb') as fin:
                    pickle.dump(y, fin)
                example_lengths.append(len(y))

    metadata_new=pd.DataFrame(data={
        'wavname':audio_ids_new,
        'emolabel':emo_labels_new,
        'genderlabel':gender_labels_new,
        'length':example_lengths
    })
    metadata_new = metadata_new.sort_values(by='length')
    metadata_new.to_csv(metadata_output_path, index=None)
    print('average y:{}'.format(np.mean(average_y)))


def filter_dataset(metapath, labels, label_name):
    data = pd.read_csv(metapath)
    columns=data.columns.values
    data['filter'] = data.apply(lambda x: 1 if x[label_name] in labels else 0, axis=1)
    data = data[data['filter'] == 1]
    data=data[columns]
    data.to_csv(metapath, index=None)
    count=data.shape[0]
    print('all data length: {}'.format(count))
    for label in labels:
        sub_count=data[data[label_name]==label].shape[0]
        print('"{}"-> {} -> {}%'.format(label,sub_count,round(sub_count/count*100,2)))


if __name__ == '__main__':
    metapath = '../data/iemocap_raw_output.csv'
    metapath_new = '../data/iemocap_raw_output_new.csv'
    audio_root_dir = '../data/raw_audio_data'
    audio_root_dir_new = '../data/raw_audio_data_new'
    label_name = 'emolabel'
    # print('bucket examples...')
    # bucket_examples(metapath, metapath_new, audio_root_dir, audio_root_dir_new)
    print('filter dataset...')
    filter_dataset(metapath_new, hparams.classes, label_name)
