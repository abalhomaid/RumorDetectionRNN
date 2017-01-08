# RumorDetectionRNN

before running any file:

1- make sure that the dataset folder is inside the project amd has name 'rumor'

![alt tag](images\sampleFileStructure.PNG)

to create folder resources:

1- run preprocessData.py main function (createTensorInput(inputFile))


to train neural network:

1- run RumorRNN.py main function 


latest result received (excluding tensorflow warnings)


Training samples: 117
Validation samples: 875
--
Training Step: 1 
| Adam | epoch: 000 | loss: 0.00000 - acc: 0.0000 -- iter: 032/117
Training Step: 2  | total loss: 0.63003
| Adam | epoch: 000 | loss: 0.63003 - acc: 0.5398 -- iter: 064/117
Training Step: 3  | total loss: 0.68456
| Adam | epoch: 000 | loss: 0.68456 - acc: 0.4836 -- iter: 096/117
Training Step: 4  | total loss: 0.69145
| Adam | epoch: 001 | loss: 0.69145 - acc: 0.5206 | val_loss: 0.69413 - val_acc: 0.4880 -- iter: 117/117
Training Step: 4  | total loss: 0.69145
| Adam | epoch: 001 | loss: 0.69145 - acc: 0.5206 | val_loss: 0.69413 - val_acc: 0.4880 -- iter: 117/117
--