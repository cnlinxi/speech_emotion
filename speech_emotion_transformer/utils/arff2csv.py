# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 11:26
# @Author  : MengnanChen
# @FileName: arff2csv.py
# @Software: PyCharm

import os
import pandas as pd


def arff2csv(arff_paths,csv_path):
    # extra features: name, class
    frame_dict = {}
    for arff_path in arff_paths:
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
                                frame_dict[line_t[1]]=[]
                            frame_dict[line_t[1]].append(float(data[index]))
                        except:
                            print('error:', line)
                            frame_dict[line_t[1]].append(data[index])
                    else:
                        if line_t[1] not in frame_dict.keys():
                            frame_dict[line_t[1]] = []
                        frame_dict[line_t[1]].append(data[index])
                    index += 1
    dataframe = pd.DataFrame(data=frame_dict)
    dataframe.to_csv(csv_path, index=None)

if __name__ == '__main__':
    arff_root_dir='../data/arff'
    arff_files=os.listdir(arff_root_dir)
    arff_paths=[os.path.join(arff_root_dir,x) for x in arff_files]
    output_path='../data/output.csv'
    arff2csv(arff_paths,output_path)
