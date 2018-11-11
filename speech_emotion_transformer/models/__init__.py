# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 17:10
# @Author  : MengnanChen
# @FileName: __init__.py
# @Software: PyCharm

from speech_emotion_transformer.models import hparams
from speech_emotion_transformer.models.speech_transformer import SpeechTransformer


def create_model(mode='train'):
    return SpeechTransformer(mode=mode,drop_rate=hparams.transformer_drop_rate)
