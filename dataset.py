# -*- coding: utf-8 -*-
# @Time    : 2018/7/13 17:18
# @Author  : MengnanChen
# @FileName: dataset.py
# @Software: PyCharm Community Edition

import os
import subprocess as sp
import itertools
import librosa

class dataset:

    def __init__(self, path, dataset_type, decode=False):
        self.dataset_type = dataset_type
        if dataset_type == "berlin":
            self.classes = {0: 'W', 1: 'L', 2: 'E', 3: 'A', 4: 'F', 5: 'T', 6: 'N'}
            self.get_berlin_dataset(path)
        elif dataset_type == "dafex":
            self.classes = {0: 'ang', 1: 'dis', 2: 'fea', 3: 'hap', 4: 'neu', 5: 'sad', 6: 'sur'}
            self.get_dafex_dataset(path, decode)

    def get_berlin_dataset(self, path):
        classes = {v: k for k, v in self.classes.items()}
        self.targets = []
        self.data = []
        for audio in os.listdir(path):
            audio_path = os.path.join(path, audio)
            y, sr = librosa.load(audio_path, sr=16000)
            self.data.append((y, sr))  # The training data is stored, and then the data is obtained by feature extraction
            # the fifth character of audio file name in berlin dataset, refer to http://emodb.bilderbar.info/index-1280.html
            self.targets.append(classes[audio[5]])

    def get_dafex_dataset(self, path, decode=False):
        males = ['4', '5', '7', '8']
        females = ['1', '2', '3', '6']
        no_audio = [3, 6]
        classes = {v: k for k, v in self.classes.items()}
        self.targets = []
        self.data = []
        self.train_sets = []
        self.test_sets = []
        get_data = True
        for speak_test in itertools.product(males, females):
            i = 0
            train = []
            test = []
            for actor_dir in os.listdir(path):
                if actor_dir[-1].isdigit():  # avoid invisible files
                    actor_path = os.path.join(path, actor_dir)
                    for block in os.listdir(actor_path):
                        if block[-1].isdigit() and int(
                                block[-1]) not in no_audio:  # avoid only video blocks and invisible files
                            block_path = os.path.join(actor_path, block)
                            for f_video in os.listdir(block_path):
                                if f_video.endswith('avi'):  # avoid invisible files
                                    ss = f_video.split()
                                    f_video = f_video.replace(" ", "\ ")  # for shell command
                                    video_path = os.path.join(block_path, f_video)
                                    audio_path = video_path.replace('.avi', '.wav')  # output
                                    if decode and get_data:
                                        sp.call("ffmpeg -i " + video_path +
                                                " -ab 160k -ac 1 -ar 22050 -vn " + audio_path, shell=True)
                                    y, sr = librosa.load(audio_path.replace("\ ", " "), sr=16000)
                                    if get_data:
                                        self.targets.append(classes[ss[6]])  # getting targets
                                        self.data.append((y, sr))  # getting signals + sr
                                    if actor_dir[-1] in speak_test:
                                        test.append(i)
                                    else:
                                        train.append(i)
                                    i = i + 1
            self.train_sets.append(train)
            self.test_sets.append(test)
            get_data = False
