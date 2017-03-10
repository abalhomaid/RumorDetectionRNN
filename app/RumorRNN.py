# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.

References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).

Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/

"""
from __future__ import division, print_function, absolute_import

import tflearn
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from app.preprocessData import *
from tflearn.data_utils import to_categorical, pad_sequences
from keras.preprocessing.sequence import pad_sequences as keraspad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, PReLU, Dropout, Embedding, LSTM, BatchNormalization, TimeDistributed

TENSOR_MODEL = r'../resources/tensorModel.model'
INPUT_DIM = 200
OUTPUT_DIM = 2
NUM_EPOCHS = 25
NUM_CLASSES = 2


def main():
    # trainModelTFLearn()
    # trainModelKeras()
    trainModelLGB()

def loadModel():
    print('not yet implemented')

    # net = tflearn.input_data([None, 100])
    # net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    # net = tflearn.lstm(net, 128, dropout=0.8)
    # net = tflearn.fully_connected(net, 2, activation='softmax')
    # net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
    #                          loss='categorical_crossentropy')
    #
    # model = tflearn.DNN(net, tensorboard_verbose=0)
    #
    # tensorModel = model.load(TENSOR_MODEL + '.meta')
    #
    # print('nice!')

def trainModelLGB():

    X_train, y_train, X_test, y_test = loadTensorInput(RUMOR_TF_INPUTPICKLE)
    model_txt_path = '../resources/LightGBMModel.txt'
    model_json_path = '../resources/LightGBMModel.json'

    X_train = keraspad_sequences(X_train, maxlen=INPUT_DIM, padding='post', value=0.)
    X_test = keraspad_sequences(X_test, maxlen=INPUT_DIM, padding='post', value=0.)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    evals_result = {}  # to record eval results for plotting

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=[lgb_train, lgb_test],
                    categorical_feature=[21],
                    evals_result=evals_result,
                    verbose_eval=10)



    print('Plot metrics during training...')
    ax = lgb.plot_metric(evals_result)
    plt.show()

    print('Plot feature importances...')
    ax = lgb.plot_importance(gbm, max_num_features=10)
    plt.show()

    print('Plot 84th tree...')  # one tree use categorical feature to split
    ax = lgb.plot_tree(gbm)
    plt.show()

    print('Save model...')
    # save model to file
    gbm.save_model(model_txt_path)

    print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

    print('Dump model to JSON...')
    # dump model to json (and save to file)
    model_json = gbm.dump_model()

    with open(model_json_path, 'w+') as f:
        json.dump(model_json, f, indent=4)

    print('Feature names:', gbm.feature_name())

    print('Calculate feature importances...')
    # feature importances
    print('Feature importances:', list(gbm.feature_importance()))




def trainModelKeras():

    trainX, trainY, testX, testY = loadTensorInput(RUMOR_TF_INPUTPICKLE)

    # Data preprocessing
    # Sequence padding
    trainX = keraspad_sequences(trainX, maxlen=INPUT_DIM, padding='post', value=0.)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    testX = keraspad_sequences(testX, maxlen=INPUT_DIM, padding='post', value=0.)
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    print('DONE:pad_sequencing')

    # Converting labels to binary vectors
    trainY = np_utils.to_categorical(trainY, nb_classes=2)
    testY = np_utils.to_categorical(testY, nb_classes=2)
    print('DONE:to_categorical')

    # # Network building
    model = Sequential()
    model.add(LSTM(output_dim=50, input_dim=1, return_sequences=False))

    model.add(Dense(784, init='normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(784, init='normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(784, init='normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(784, init='normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(784, init='normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, nb_epoch=NUM_EPOCHS)
    model.save(TENSOR_MODEL)


    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print('success')

def trainModelTFLearn():

    # Python 2
    trainX, trainY, testX, testY = loadTensorInput(RUMOR_TF_INPUTPICKLE)

    # Data preprocessing
    # Sequence padding
    print(type(trainX))

    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)

    print('DONE:pad_sequencing')

    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    print('DONE:to_categorical')

    # # Network building
    # with tf.device('/gpu:0'):
    net = tflearn.input_data([None, 100])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=32, n_epoch=1)

    model.save(TENSOR_MODEL)
    print('done')

def printLargestSmallestNumElements(list):
    largest = 0
    smallest = 100000
    i = 0
    j = 0
    for idx, x in enumerate(list):
        if len(x) > largest:
            largest = len(x)
            i = idx

        if len(x) < smallest:
            smallest = len(x)
            j = idx

    print('largest number of elements:', largest)
    print('index:', i)
    print('--------------------')
    print('smallest number of elements:', smallest)
    print('index:', j)


if __name__ == "__main__": main()
