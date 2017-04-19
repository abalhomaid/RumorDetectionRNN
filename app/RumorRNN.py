from __future__ import division, print_function, absolute_import

from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Embedding, Dense, PReLU, Dropout, regularizers
from keras.layers import LSTM
from keras.models import load_model, Sequential
from keras.utils import plot_model
from tflearn.data_utils import to_categorical

from app.preprocessData import *

MODEL_PATH = r'../weight_adam/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
MODEL_PATH_SGD = r'../weight_sgd/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
MODEL_PATH_ADAGRAD = r'../weight_adagrad/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
INPUT_DIM = 200
OUTPUT_DIM = 2
NUM_EPOCHS = 25
NUM_CLASSES = 2
RUMOR_TF_INPUTPICKLE = r'../resources/tensorInput.pkl'

def loadInput(inputFile):
    # load pickle
    with open(inputFile, "rb") as input:
        print('loading input\n')

        outputList = pickle.load(input)
        # X_train, y_train = zip(*outputList)
        X_train, y_train = outputList

        outputList = pickle.load(input)
        # X_test, y_test = zip(*outputList)
        X_test, y_test = outputList

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        return X_train, y_train, X_test, y_test

def main():
    trainEmbedding()
    # plot_keras_model()

def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=20000, output_dim=EMBEDDING_DIM))
    model.add(Dropout(0.4))
    # model.add(Embedding(input_dim=MAX_SEQUENCE_LENGTH + 1, output_dim=EMBEDDING_DIM))
    # model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dropout(0.4))

    model.add(Dense(784, kernel_regularizer=regularizers.l2(0.07)))
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(784, kernel_regularizer=regularizers.l2(0.07)))
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))
    return model

""" trains on X using Keras's embedding layer (X is list of word IDs)"""
def trainEmbedding():
    X_train, y_train, X_test, y_test = loadInput(RUMOR_TF_INPUTPICKLE)

    y_train = to_categorical(y_train, nb_classes=2)
    y_test = to_categorical(y_test, nb_classes=2)

    model = build_model()

    # model.load_weights('../weight_adam/weights.06-0.86.hdf5')
    model.load_weights('../weight_sgd/weights.07-0.70-0.51.hdf5')
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-8, decay=0.)
    model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint(MODEL_PATH_ADAGRAD, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=False, write_images=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[tensor_board, model_checkpoint])



def plot_keras_model():
    model = load_model(MODEL_PATH)
    plot_model(model, to_file='model.png')

if __name__ == "__main__": main()
