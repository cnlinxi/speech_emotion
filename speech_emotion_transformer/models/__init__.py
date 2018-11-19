# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 17:10
# @Author  : MengnanChen
# @FileName: __init__.py
# @Software: PyCharm

from models import hparams
from models.speech_transformer import SpeechTransformer
from models.MLP import MLP


def create_model(mode='train',model_type='transformer'):
    if model_type=='transformer':
        return SpeechTransformer(mode=mode,drop_rate=hparams.transformer_drop_rate)
    elif model_type=='mlp':
        return MLP(mode,hparams.mlp_dropout_rate)
