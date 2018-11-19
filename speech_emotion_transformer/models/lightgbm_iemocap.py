# -*- coding: utf-8 -*-
# @Time    : 2018/11/18 10:20
# @Author  : MengnanChen
# @FileName: lightgbm_iemocap.py
# @Software: PyCharm

import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from models import hparams

print('load data...')

df_data = pd.read_csv(hparams.emo_opensmile_data_path)
filter_columns = ['name', 'emotion', 'genderlabel', 'wavname']
all_columns = [x for x in df_data.columns if x not in filter_columns]
label_column_name = 'emolabel'
lbl = LabelEncoder()
df_data[label_column_name] = lbl.fit_transform(df_data[label_column_name])
print('classes: {}'.format(list(lbl.classes_)))
n_classes=len(list(lbl.classes_))
feature_column_names = [x for x in all_columns if x != label_column_name]

X_train, X_test, y_train, y_test = train_test_split(df_data[feature_column_names], df_data[label_column_name],
                                                    test_size=0.1, random_state=2018)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': n_classes,
    'early_stopping': 30,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
}

print('start training...')
gbm = lgb.train(params,
                num_boost_round=5000,
                train_set=lgb_train,
                valid_sets=lgb_train)
y_pred = gbm.predict(X_test)  # 2-D array
y_pred=np.argmax(y_pred,axis=-1)

acc = accuracy_score(y_test, y_pred)
print('eval acc: {:4f}'.format(acc))

print('feature importance:')
feature_importances=dict(zip(feature_column_names,gbm.feature_importance()))
print(sorted(feature_importances.items(), key=lambda d: d[1],reverse=True)[:10])

print('save model...')
os.makedirs(hparams.lgb_model_save_dir, exist_ok=True)
gbm.save_model(os.path.join(hparams.lgb_model_save_dir, 'lgb_debug3.model'))
