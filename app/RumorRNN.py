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

MODEL_PATH = r'../resources/my_model.h5'
INPUT_DIM = 200
OUTPUT_DIM = 2
NUM_EPOCHS = 25
NUM_CLASSES = 2


def main():
    # trainModelKeras()
    # trainModelLGB()
    trainEmbedding()

""" trains on X using Microsoft's LightGBM"""
def trainModelLGB():

    X_train, y_train, X_test, y_test = loadInput(RUMOR_TF_INPUTPICKLE)
    model_txt_path = '../resources/LightGBMModel.txt'
    model_json_path = '../resources/LightGBMModel.json'

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

    print('Plot tree...')  # one tree use categorical feature to split
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


""" trains on X using Keras's embedding layer (X is list of word IDs)"""
def trainEmbedding():

    MAX_SEQUENCE_LENGTH = 1000
    EMBEDDING_DIM = 100
    X_train, y_train, X_test, y_test = loadInput(RUMOR_TF_INPUTPICKLE)

    y_train = to_categorical(y_train, nb_classes=2)
    y_test = to_categorical(y_test, nb_classes=2)

    # create the model
    model = Sequential()
    model.add(Embedding(input_dim=MAX_SEQUENCE_LENGTH, output_dim=EMBEDDING_DIM))
    model.add(LSTM(100))

    model.add(Dense(784, init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))

    model.add(Dense(784, init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))

    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=20, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # Save model
    model.save(MODEL_PATH)

""" trains on X using Keras (X is list of word tfidfs), not yet tested"""
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

if __name__ == "__main__": main()
