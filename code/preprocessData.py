import json
import re
import logging
import os
import pickle
import sys
from os.path import basename
from gensim import corpora, models, utils
from gensim.corpora import MmCorpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


CORPUS_FILE_PATH = r'../resources/rumorCorpus.mm'
DICTIONARY_FILE_PATH = r'../resources//rumorDictionary.dict'
DOC2VEC_FILE_PATH = r'../resources/rumorModelDoc2Vec.bin'
INPUT_PATH = r'../rumor/twitter_json'
RUMOR_TF_INPUTJSON = r'../resources/tensorInput.json'
RUMOR_TF_INPUTPICKLE = r'../resources/tensorInput.pkl'
TEST_SET_FILE_PATH = r'../resources/testSet_twitter.txt'
TFIDF_FILE_PATH = r'../resources/rumorTfidf.tfidf_model'
TRAIN_SET_FILE_PATH = r'../resources/testSet_twitter.txt'
TWITTER_LABEL_PATH = r'../resources/twitter_label.txt'
WORD2VEC_FILE_PATH = r'../resources/rumorModelWord2Vec.txt'


def main():
    createTensorInput(RUMOR_TF_INPUTPICKLE)



def loadDictionary(dictionaryPath):
    return corpora.Dictionary.load(dictionaryPath)

def loadCorpus(corpusPath):
    return MmCorpus(corpusPath)

def loadTfidf(tfidfPath):
    return models.TfidfModel.load(tfidfPath)

"""load all labels (true/false) for training and test set into LABEL_DICT"""
def loadLabels(labelPath):
    LABEL_DICT = {}
    with open(labelPath) as infile:
        for line in infile:
            line = line.split('\t')
            # print(line)
            key = line[1].split(':')[1]
            value = line[0].split(':')[1]
            LABEL_DICT[key] = value

    return LABEL_DICT


def loadTensorInput(inputFile):
    # load pickle
    with open(inputFile, "rb") as input:
        i = 0

        print('loading input\n')

        # PICKLE load

        outputList = pickle.load(input)
        trainX, trainY = zip(*outputList)

        outputList = pickle.load(input)
        testX, testY = zip(*outputList)

        trainX = list(trainX)
        trainY = list(trainY)
        testX = list(testX)
        testY = list(testY)

        for idx, i in enumerate(trainX):
            trainX[idx] = trainX[idx].values()
        for idx, i in enumerate(trainY):
            trainY[idx] = trainY[idx].values()
        for idx, i in enumerate(testX):
            testX[idx] = testX[idx].values()
        for idx, i in enumerate(testY):
            testY[idx] = testY[idx].values()

        print('Success')

    return trainX, trainY, testX, testY

def createTensorInput(inputFile):
    # create corpus from tweets
    # create idf from corpus
    # write idf, ground truth to file, using pickle

    trainSetList = set()
    testSetList = set()

    # load train filenames into list
    with open(TRAIN_SET_FILE_PATH, 'rb') as f:
        for line in f:
            trainSetList.add(line.rstrip())

    # load test filenames into list
    with open(TEST_SET_FILE_PATH, 'rb') as f:
        for line in f:
            testSetList.add(line.rstrip())

    with open(inputFile, "wb") as output:
        LABEL_DICT = loadLabels(TWITTER_LABEL_PATH)
        trainOutputList = []
        testOutputList = []
        for fname in os.listdir(INPUT_PATH):
            corpus = createCorpus(os.path.join(INPUT_PATH, fname))
            tfidf = createTfidf(corpus)
            line = tfidf.idfs, {'label': LABEL_DICT[fname.split('.')[0]]}


            if fname in trainSetList:
                trainOutputList.append(line)
            else:
                testOutputList.append(line)

            # write data json
            # json.dumps(line, output)

        # write data pickle
        pickle.dump(trainOutputList, output)
        pickle.dump(testOutputList, output)

def createTfidf(corpus, tfidfSavePath = None):
    tfidf = models.TfidfModel(corpus)

    if tfidfSavePath is not None:
        tfidf.save(tfidfSavePath)

    return tfidf

def createCorpus(inputPath):
    # Create corpus from inputPath
    # text = rumorTextCorpus(INPUT_PATH)
    text = corpora.TextCorpus(inputPath)

    # Uncomment to save dictionary
    # text.dictionary.save(DICTIONARY_FILE_PATH)

    # Uncomment to save corpus
    # MmCorpus.serialize(CORPUS_FILE_PATH, text)

    return text

def createCorpusSingleFile(inputPath, dictionarySavePath, corpusSavePath):
    text = rumorTextCorpus(inputPath)
    text.dictionary.save(dictionarySavePath)
    # save corpus
    MmCorpus.serialize(corpusSavePath, text)

def createWord2Vec():
    sentences = MySentences(INPUT_PATH)  # a memory-friendly iterator
    print('SENTENCES:', sentences)
    model = models.Word2Vec(sentences)
    model.save_word2vec_format(WORD2VEC_FILE_PATH, None, False)

def createDoc2Vec():
    model = models.Doc2Vec.load_word2vec_format(WORD2VEC_FILE_PATH)
    model.save(DOC2VEC_FILE_PATH)


def testTfidfInput():
    # one file
    # don't need id -> word
    # create corpus from tweets
    # create idf from corpus
    # write idf, ground truth to file, using pickle
    inputPath = r'/home/ubuntu/Desktop/Tensorflow/datasets/rumorTest/twitter_json'
    samplePath = r'/home/ubuntu/Desktop/Tensorflow/datasets/rumorTest/twitter_json/Airfance.json'
    dictionarySavePath = r'/home/ubuntu/Desktop/Tensorflow/datasets/rumorTest/dict.dict'
    corpusSavePath = r'/home/ubuntu/Desktop/Tensorflow/datasets/rumorTest/corpus.mm'

    createCorpusSingleFile(inputPath, dictionarySavePath, corpusSavePath)
    corpus = loadCorpus(corpusSavePath)

    tfidfSavePath = r'/home/ubuntu/Desktop/Tensorflow/datasets/rumorTest/tfidf.tfidf_model'
    tfidf = createTfidf(corpus, tfidfSavePath)

    print(tfidf.idfs[1])
    print(basename(samplePath))

    # to get filename without extension
    # basename(os.path.splitext(samplePath)[0])

    line = tfidf.idfs, {'label': LABEL_DICT['Airfrance']}

    # write data
    output = open(r'/home/ubuntu/Desktop/Tensorflow/datasets/rumorTest/output.pkl', 'wb')
    pickle.dump(line, output)
    output.close()

    # read data
    output = open(r'/home/ubuntu/Desktop/Tensorflow/datasets/rumorTest/output.pkl', 'rb')
    read = pickle.load(output)
    print(read)

    print(tfidf)


    # for line in open(r'../resources/twitter_json/Airfrance.json'):
    #     line = re.sub(' "source":(.[^,]+)",', '', line)  # remove json.loads corrupters
    #     print(line)



class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                line = re.sub(' "source":(.[^,]+)",', '', line) # remove json.loads corrupters
                # print('LINEAFTER:' + line)
                jsonObject = json.loads(line)
                yield jsonObject['text'].split()

class rumorTextCorpus(corpora.TextCorpus):

    def __init__(self, dirname):
        self.dirname = dirname
        super(rumorTextCorpus, self).__init__(dirname)


    def get_texts(self):
        for fname in os.listdir(self.dirname):
            self.input = os.path.join(self.dirname, fname)
            with self.getstream() as lines:
                for lineno, line in enumerate(lines):
                    line = re.sub(' "source":(.[^,]+)",', '', line)  # remove json.loads corrupters
                    # print('LINEAFTER:' + line)
                    jsonObject = json.loads(line)
                    if self.metadata:
                        yield utils.tokenize(jsonObject['text'], lowercase=True), (lineno,)
                    else:
                        yield utils.tokenize(jsonObject['text'], lowercase=True)





if __name__ == "__main__": main()
