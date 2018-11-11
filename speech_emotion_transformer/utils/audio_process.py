# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 15:08
# @Author  : MengnanChen
# @FileName: audio_process.py
# @Software: PyCharm


import os
import subprocess
from six.moves import cPickle as pickle

import numpy as np
import pandas as pd
import librosa


class AudioProcess(object):
    def __init__(self):
        self.open_smile_root_dir = r'D:\ProgramFile\Program\openSMILE-2.1.0'
        self.open_smile_path = os.path.join(self.open_smile_root_dir, 'bin\Win32')
        self.config_path = os.path.join(self.open_smile_root_dir, 'config\emo_large.conf')
        self.output_arff_dir = '../data/arff'
        self.output_csv_path = '../data/iemocap_output_emo_large.csv'
        self.raw_output_dir='../data/raw_audio_data'

    def arff2csv(self, arff_paths, csv_path):
        # extra features: name, class
        frame_dict = {}
        for arff_path in arff_paths:
            print('process arff2csv, arff path:{}'.format(arff_path))
            with open(arff_path, 'rb') as fin:
                lines = fin.readlines()
                lines = [x.decode('utf-8').strip('\r\n ') for x in lines]
                data = ''
                index = 1
                while data == '':
                    data = lines[-index].strip('\r\n ')
                    index += 1
                data = data.split(',')
                index = 0
                for line in lines:
                    line_t = line.split(' ')
                    if line_t[0] == '@attribute':
                        if line_t[2] == 'numeric':
                            try:
                                if line_t[1] not in frame_dict.keys():
                                    frame_dict[line_t[1]] = []
                                frame_dict[line_t[1]].append(float(data[index]))
                            except:
                                # print('error:', line)
                                frame_dict[line_t[1]].append(data[index])
                        else:
                            if line_t[1] not in frame_dict.keys():
                                frame_dict[line_t[1]] = []
                            frame_dict[line_t[1]].append(data[index])
                        index += 1
        dataframe = pd.DataFrame(data=frame_dict)
        dataframe.to_csv(csv_path, index=None)

    def audio2vec(self, audio_path):
        audio_path = os.path.abspath(audio_path)
        config_path = os.path.abspath(self.config_path)
        output_arff_path = os.path.abspath(
            os.path.join(self.output_arff_dir, '{}.arff'.format(os.path.basename(audio_path).split('.')[0])))
        # cmd='SMILExtract_Release -C {} -I {} -O {}'.format(config_path,audio_path,output_arff_path)
        subprocess.run(
            [r'{}\SMILExtract_Release'.format(self.open_smile_path), '-C', config_path, '-I', audio_path, '-O',
             output_arff_path],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

    def process_iemocap(self, iemocap_root_dir):
        sessions = [os.path.join(iemocap_root_dir, x) for x in os.listdir(iemocap_root_dir) if x.startswith('Session')]
        wav_names = []

        emo_labels = []
        gender_labels = []
        for session in sessions:
            print('process {}...'.format(session))
            wav_root_dir = os.path.join(session, 'sentences/wav')
            wav_dirs = [os.path.join(wav_root_dir, x) for x in os.listdir(wav_root_dir)]
            for wav_dir in wav_dirs:
                print('process wav, wav dir:{}'.format(wav_dir))
                wav_dir_name = os.path.basename(wav_dir)
                emo_meta_file_path = os.path.join(session, 'dialog/EmoEvaluation', '{}.txt'.format(wav_dir_name))
                with open(emo_meta_file_path, 'rb') as fin:
                    for line in fin:
                        line = line.decode('utf-8').strip('\r\n ')
                        if not line or line[0] != '[':
                            continue
                        wav_file_name = '{}.wav'.format(line.split('\t')[1])
                        wav_file_path = os.path.join(wav_dir, wav_file_name)

                        gender_label = wav_file_name.split('_')[2][0]
                        gender_labels.append(gender_label)
                        emo_label = line.split('\t')[2]
                        emo_labels.append(emo_label)
                        self.audio2vec(wav_file_path)
                        wav_names.append(wav_file_name.split('.')[0])

        arff_paths = [os.path.join(self.output_arff_dir, x) for x in os.listdir(self.output_arff_dir)]
        self.arff2csv(arff_paths, self.output_csv_path)
        dataframe = pd.read_csv(self.output_csv_path)
        dataframe['emolabel'] = emo_labels
        dataframe['genderlabel']=gender_labels
        dataframe['wavname'] = wav_names
        dataframe.to_csv(self.output_csv_path, index=None)

    def raw_iemocap2vec(self):
        sessions = [os.path.join(iemocap_root_dir, x) for x in os.listdir(iemocap_root_dir) if x.startswith('Session')]
        if not os.path.exists(self.raw_output_dir):
            os.makedirs(self.raw_output_dir,exist_ok=True)
        with open(self.output_csv_path,'wb') as fout:
            fout.write('{},{},{},{}\n'.format('wavname','emolabel','genderlabel','pikpath').encode('utf-8'))
            for session in sessions:
                print('process {}...'.format(session))
                wav_root_dir = os.path.join(session, 'sentences/wav')
                wav_dirs = [os.path.join(wav_root_dir, x) for x in os.listdir(wav_root_dir)]
                for wav_dir in wav_dirs:
                    print('process wav, wav dir:{}'.format(wav_dir))
                    wav_dir_name = os.path.basename(wav_dir)
                    emo_meta_file_path = os.path.join(session, 'dialog/EmoEvaluation', '{}.txt'.format(wav_dir_name))
                    with open(emo_meta_file_path, 'rb') as fin:
                        for line in fin:
                            line = line.decode('utf-8').strip('\r\n ')
                            if not line or line[0] != '[':
                                continue
                            wav_file_name = '{}.wav'.format(line.split('\t')[1])
                            wav_file_path = os.path.join(wav_dir, wav_file_name)

                            gender_label = wav_file_name.split('_')[2][0]
                            emo_label = line.split('\t')[2]
                            y,sr=librosa.load(wav_file_path,sr=16000)
                            raw_audio_pik_path=os.path.join(self.raw_output_dir,'{}.pik'.format(wav_file_name))
                            with open(raw_audio_pik_path,'wb') as fout_raw:
                                pickle.dump((y,sr),fout_raw)
                            audio_data='{},{},{},{}\n'.format(wav_file_name.split('.')[0],emo_label,gender_label,raw_audio_pik_path).encode('utf-8')
                            fout.write(audio_data)


if __name__ == '__main__':
    iemocap_root_dir = r'D:/Download/Compressed/IEMOCAP_full_release'
    audio_processor = AudioProcess()
    # audio_processor.process_iemocap(iemocap_root_dir)
    audio_processor.raw_iemocap2vec()
