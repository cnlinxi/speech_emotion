#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 21:32
# @Author  : MengnanChen
# @Site    : 
# @File    : train.py
# @Software: PyCharm Community Edition

import sys
import os
sys.path.append(os.path.join(os.getcwd(),'utility'))

from utility import functions, globalvars
import dataset

from keras.layers import Input, Dense, Masking, Dropout, LSTM, Bidirectional, Activation
from keras.layers.merge import dot
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from six.moves import cPickle
import numpy as np
import os

def get_data():
    print("Loading data and features...")
    db = cPickle.load(open(globalvars.dataset + '_db.p', 'rb'))
    f_global = cPickle.load(open(globalvars.dataset + '_features.p', 'rb'))

    nb_samples = len(db.targets)
    print("Number of samples: " + str(nb_samples))

    y = np.array(db.targets)
    y = to_categorical(y, num_classes=globalvars.nb_classes)

    x_train, x_test, y_train, y_test = train_test_split(f_global, y, test_size=0.30, random_state=2018)

    u_train = np.full((x_train.shape[0], globalvars.nb_attention_param),
                      globalvars.attention_init_value, dtype=np.float64)
    u_test = np.full((x_test.shape[0], globalvars.nb_attention_param),
                     globalvars.attention_init_value, dtype=np.float64)

    return u_train, x_train, y_train, u_test, x_test, y_test

def create_model(u_train, x_train, y_train, u_test, x_test, y_test):
    # Logistic regression for learning the attention parameters with a standalone feature as input
    input_attention = Input(shape=(globalvars.nb_attention_param,))
    u = Dense(globalvars.nb_attention_param, activation='softmax')(input_attention)

    # Bi-directional Long Short-Term Memory for learning the temporal aggregation
    # Input shape: (time_steps, features,)
    input_feature = Input(shape=(globalvars.max_len, globalvars.nb_features))
    x = Masking(mask_value=-100.0)(input_feature)

    x = Dense(globalvars.nb_hidden_units, activation='relu')(x)
    x = Dropout(globalvars.dropout_rate)(x)

    x = Dense(globalvars.nb_hidden_units, activation='relu')(x)
    x = Dropout(globalvars.dropout_rate)(x)

    y = Bidirectional(LSTM(globalvars.nb_lstm_cells, return_sequences=True,
                           dropout=globalvars.dropout_rate))(x)

    # To compute the final weights for the frames which sum to unity
    alpha = dot([u, y], axes=-1)
    alpha = Activation('softmax')(alpha)

    # Weighted pooling to get the utterance-level representation
    z = dot([alpha, y], axes=1)

    # Get posterior probability for each emotional class
    output = Dense(globalvars.nb_classes, activation='softmax')(z)

    model = Model(inputs=[input_attention, input_feature], outputs=output)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()

    globalvars.globalVar += 1

    # file_path = 'weights_blstm_hyperas_' + str(globalvars.globalVar) + '.h5'
    file_path = 'weights_blstm_hyperas.h5'
    callback_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=1,
            mode='auto'
        ),
        ModelCheckpoint(
            filepath=file_path,
            monitor='val_acc',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]

    hist = model.fit([u_train, x_train], y_train, batch_size=128, epochs=200, verbose=2,
                     callbacks=callback_list, validation_data=([u_test, x_test], y_test))
    h = hist.history
    acc = np.asarray(h['acc'])
    loss = np.asarray(h['loss'])
    val_loss = np.asarray(h['val_loss'])
    val_acc = np.asarray(h['val_acc'])

    acc_and_loss = np.column_stack((acc, loss, val_acc, val_loss))
    save_file_blstm = 'blstm_run_' + str(globalvars.globalVar) + '.txt'
    with open(save_file_blstm, 'w'):
        np.savetxt(save_file_blstm, acc_and_loss)

    score, accuracy = model.evaluate([u_test, x_test], y_test, batch_size=128, verbose=1)
    print('Final validation accuracy: %s' % accuracy)

    return {'accuracy': accuracy, 'model': model}

if __name__ == '__main__':
    # if dataset is not exists, generate them.
    if (not os.path.exists('berlin_db.p')) or (not os.path.exists('berlin_features.p')):
        dataset_type=globalvars.dataset
        if globalvars.load_data:
            ds = dataset.dataset(path=globalvars.train_data_path, dataset_type=dataset_type)
            print('Writing ' + dataset_type + ' data set to file...')
            cPickle.dump(ds, open(dataset_type + '_db.p', 'wb'))
        else:
            print('Loading data from ' + dataset_type + ' data set...')
            ds = cPickle.load(open(dataset_type + '_db.p', 'rb'))

        if globalvars.feature_extract:
            functions.feature_extract(ds.data, ds.dataset_type, nb_samples=len(ds.targets), dataset=ds)

    try:
        U_train, X_train, Y_train, U_test, X_test, Y_test = get_data()
        result_dict=create_model(U_train,X_train,Y_train,U_test,X_test,Y_test) # define model & train
        print('the result is:{}'.format(str(result_dict))) # the model has saved in filename weights_blstm_hyperas_....h5

        ##### Evaluation
        # to be deleted...
        best_model_idx = 1
        best_score = 0.0
        for i in range(1, (globalvars.globalVar + 1)):
            print('Evaluate models:')

            # load model
            model_path = 'weights_blstm_hyperas_' + str(i) + '.h5'
            model = load_model(model_path)

            scores = model.evaluate([U_test, X_test], Y_test)
            if (scores[1] * 100) > best_score:
                best_score = (scores[1] * 100)
                best_model_idx = i

            print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

        print('The best model is weights_blstm_hyperas_' + str(best_model_idx) + '.h5')
        #####
    except IOError:
        print('No training data found')