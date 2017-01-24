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

from app.preprocessData import *
from tflearn.data_utils import to_categorical, pad_sequences
from keras.preprocessing.sequence import pad_sequences as keraspad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, PReLU, Dropout, Embedding, LSTM

TENSOR_MODEL = r'../resources/tensorModel.model'



def main():
    # trainModelTFLearn()
    trainModelKeras()

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


def trainModelKeras():

    # Python 2
    trainX, trainY, testX, testY = loadTensorInput(RUMOR_TF_INPUTPICKLE)
    
    # Data preprocessing
    # Sequence padding
    trainX = keraspad_sequences(trainX, maxlen=100, padding='post', value=0.)
    testX = keraspad_sequences(testX, maxlen=100, padding='post', value=0.)

    print('DONE:pad_sequencing')

    # Converting labels to binary vectors
    trainY = np_utils.to_categorical(trainY, nb_classes=2)
    testY = np_utils.to_categorical(testY, nb_classes=2)

    print('DONE:to_categorical')

    # # Network building
    model = Sequential()
    model.add(Embedding(input_dim = 100, output_dim = 128))
    model.add(LSTM(output_dim = 100))
    model.add(Dense(output_dim = 2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, validation_data=(testX, testY), verbose=2, batch_size=32, nb_epoch=5)
    model.save(TENSOR_MODEL)
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
