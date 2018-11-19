# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 20:52
# @Author  : MengnanChen
# @FileName: mlp_data_mod_utils.py
# @Software: PyCharm

import sys
sys.path.append('../')

import pandas as pd

from models import hparams


def filter_dataset(data_path, output_path):
    data = pd.read_csv(data_path)
    filter_columns = [x for x in data.columns if x not in ['name', 'emotion', 'genderlabel', 'wavname']]
    data['filter'] = data.apply(lambda x: 1 if x['emolabel'] in hparams.classes else 0,axis=1)
    data = data[data['filter'] == 1]
    data[filter_columns].to_csv(output_path,index=None)


if __name__ == '__main__':
    data_path = '../data/iemocap_output_emo_large.csv'
    output_path = '../data/iemocap_output_emo_large_new.csv'
    filter_dataset(data_path, output_path)
