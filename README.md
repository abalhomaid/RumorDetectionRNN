# RumorDetectionRNN

## Libraries used

1. gensim
2. keras
3. pickle

## Algorithm

### Dataset pre-process (preprocessData.py)

1. for each train file f in twitter_json, 
    1. put value of 'text' key in a list X_train, do this for all lines
    2. Tokenize X_train
    3. Convert each text in X_train to sequences
    4. pad X_train
2. dump X_train and Y_train using pickle
3. do step 1 and 2 for test every test file

### Training neural network

1. load trainX, trainY, testX, testY using <code>loadTensorInput()</code> Each item is now a list of list
2. categorize trainY and testY to two classes
3. build the neural net model using tflearn (LSTM RNN)
    1. activation='softmax'
    2. optimizer='adam'
    3. learning_rate=0.001
    4. loss='categorical_crossentropy'
4. fit the model
    1. n_epoch=20
5. save the model

## before running any file:

1. make sure that the dataset folder is inside the project and has name 'rumor'

![alt tag](images/sampleFileStructure.PNG)

## To create folder resources:

1. run preprocessData.py


## To train neural network:

1. run RumorRNN.py

## Result

Train on 875 samples, validate on 118 samples

20 epochs

<code>
875/875 [==============================] - 15s - loss: 0.5089 - acc: 0.7646 - val_loss: 0.4852 - val_acc: 0.7542
Epoch 13/20
875/875 [==============================] - 15s - loss: 0.3863 - acc: 0.8274 - val_loss: 0.7699 - val_acc: 0.7203
Epoch 14/20
875/875 [==============================] - 15s - loss: 0.2909 - acc: 0.8720 - val_loss: 0.8753 - val_acc: 0.7373
Epoch 15/20
875/875 [==============================] - 15s - loss: 0.1825 - acc: 0.9314 - val_loss: 1.3211 - val_acc: 0.7119
Epoch 16/20
875/875 [==============================] - 15s - loss: 0.1228 - acc: 0.9543 - val_loss: 1.5710 - val_acc: 0.6695
Epoch 17/20
875/875 [==============================] - 15s - loss: 0.0728 - acc: 0.9794 - val_loss: 2.1107 - val_acc: 0.6525
Epoch 18/20
875/875 [==============================] - 15s - loss: 0.0792 - acc: 0.9749 - val_loss: 2.3427 - val_acc: 0.6695
Epoch 19/20
875/875 [==============================] - 15s - loss: 0.0710 - acc: 0.9783 - val_loss: 2.8942 - val_acc: 0.6356
Epoch 20/20
875/875 [==============================] - 15s - loss: 0.0675 - acc: 0.9794 - val_loss: 1.9913 - val_acc: 0.6695
118/118 [==============================] - 0s
Accuracy: 66.95%
</code>
Process finished with exit code 0
