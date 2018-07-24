# -*- coding: utf-8 -*-
# @Time    : 2018/7/10 10:46
# @Author  : MengnanChen
# @FileName: predict.py
# @Software: PyCharm Community Edition

import sys
import os
sys.path.append(os.path.join(os.getcwd(),'utility'))

from keras.models import load_model
from utility import functions, globalvars
import librosa
import numpy as np

emotion_classes=['anger','boredom','disgust','anxiety/fear','happiness','sadness','neutral']

def predict(data_path:str,model_path:str):
    y,sr=librosa.load(data_path,sr=16000) # librosa:load wav
    f=functions.feature_extract_test(data=(y,sr)) # feature extraction
    u=np.full((f.shape[0],globalvars.nb_attention_param),globalvars.attention_init_value,dtype=np.float64)
    model=load_model(model_path)
    # the shape of result is [1,7], e.g.:[[0.31214175 0.04727687 0.01413498 0.13356456 0.4746141  0.00477368 0.01349405]]
    result=model.predict([u,f])
    # print('type of result:',type(result)) # <class 'numpy.ndarray'>
    return result[0]

if __name__ == '__main__':
    model_path='weights_blstm_hyperas_1.h5'
    data_path='data/test/201_happy.wav'
    result=predict(data_path,model_path)
    assert len(result)==globalvars.nb_classes
    index_top_n=np.argsort(result)[-globalvars.top_n:]
    human_result=[emotion_classes[i] for i in index_top_n]
    probability_result=[result[i] for i in index_top_n]
    result=zip(human_result,probability_result)
    for x in result:
        print('the top {} emotion is:{}'.format(globalvars.top_n,x))