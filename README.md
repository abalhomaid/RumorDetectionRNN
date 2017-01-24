# RumorDetectionRNN

## Libraries used

1. gensim
2. tflearn
3. keras
4. pickle

## Algorithm

### Dataset pre-process (preprocessData.py)

1. for each train file f in twitter_json, 
    1. put value of 'text' key in a list l, do this for all lines
    2. tfidf = compute tf*idf with l as input
    3. create tuple t (tfidf, ground_truth of f)
2. pickle.dump all tuples
3. do step 1 and 2 for test every test file

### Training neural network

1. load trainX, trainY, testX, testY using <code>loadTensorInput()</code> Each item is now a list of list
2. pad trainX and testX so that each item has the same length (currently 100 because of machine capability but should be around 8000)  
3. categorize trainY and testY to two classes
4. build the neural net model using tflearn (LSTM RNN)
    1. activation='softmax'
    2. optimizer='adam'
    3. learning_rate=0.001
    4. loss='categorical_crossentropy'
5. fit the model
    1. n_epoch=1, due to machine capability

## before running any file:

1. make sure that the dataset folder is inside the project amd has name 'rumor'

![alt tag](images/sampleFileStructure.PNG)

## To create folder resources:

1. run preprocessData.py main function (createTensorInput(inputFile))


## To train neural network:

1. run RumorRNN.py main function 



## Result

Train on 875 samples, validate on 117 samples

<code>
Training samples: 117  
Validation samples: 875  

7s - loss: 0.6813 - acc: 0.5497 - val_loss: 0.9414 - val_acc: 0.2735
Epoch 2/5
2s - loss: 0.6695 - acc: 0.5817 - val_loss: 0.7839 - val_acc: 0.2393
Epoch 3/5
2s - loss: 0.6664 - acc: 0.5840 - val_loss: 0.9430 - val_acc: 0.2650
Epoch 4/5
2s - loss: 0.6634 - acc: 0.5863 - val_loss: 0.8857 - val_acc: 0.2650
Epoch 5/5
2s - loss: 0.6620 - acc: 0.5874 - val_loss: 0.9371 - val_acc: 0.2735
</code>
