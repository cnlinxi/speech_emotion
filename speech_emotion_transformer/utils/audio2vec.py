# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 13:52
# @Author  : MengnanChen
# @FileName: audio2vec.py
# @Software: PyCharm

import os
import subprocess


def audio2vec(audio_path,output_arff_dir,config_path):
    audio_path= os.path.abspath(audio_path)
    config_path=os.path.abspath(config_path)
    output_arff_path=os.path.abspath(os.path.join(output_arff_dir,'{}.arff'.format(os.path.basename(audio_path).split('.')[0])))
    # cmd='SMILExtract_Release -C {} -I {} -O {}'.format(config_path,audio_path,output_arff_path)
    subprocess.Popen([r'C:\program\openSMILE-2.1.0\bin\Win32\SMILExtract_Release','-C',config_path,'-I',audio_path,'-O',output_arff_path],
                     shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)

if __name__ == '__main__':
    audio_path='../data/audio/00001.wav'
    output_arff_dir='../data/arff'
    config_path=r'C:\program\openSMILE-2.1.0\config\IS09_emotion.conf'
    audio2vec(audio_path,output_arff_dir,config_path)
