import datetime
import json
import re
import logging
import os
import pickle
import numpy as np

from gensim import corpora, utils
from gensim.corpora import MmCorpus
from dateutil import parser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


CORPUS_FILE_PATH = r'../resources/rumorCorpus.mm'
DICTIONARY_FILE_PATH = r'../resources/rumorDictionary.pkl'
DATA_PATH = r'../rumor/twitter_json'
RUMOR_TF_INPUTPICKLE = r'../resources/tensorInput.pkl'
TEST_SET_FILE_PATH = r'../rumor/testSet_twitter.txt' # given file
TRAIN_SET_FILE_PATH = r'../rumor/trainSet_twitter.txt' # given file
TWITTER_LABEL_PATH = r'../rumor/twitter_label.txt' # given file

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100


def main():
    # createInputKeras(RUMOR_TF_INPUTPICKLE)
    # loadInput(RUMOR_TF_INPUTPICKLE)
    # createInput(RUMOR_TF_INPUTPICKLE)
    E = getSequenceFromFile('../rumor/twitter_json/Airfrance.json')
    time_series = variable_length_time_series(E, N=100)
    test = create_tfidfs(time_series)
    print('Done preprocessing.')

"""load all labels (true/false) for training and test set into LABEL_DICT"""
"""creates a dictionary Filename: label"""
def loadLabels(labelPath):
    LABEL_DICT = {}
    with open(labelPath) as infile:
        for line in infile:
            line = line.split('\t')
            print(line)
            key = line[1].split(':')[1]
            value = line[0].split(':')[1]
            LABEL_DICT[key] = value

    return LABEL_DICT

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


""" Creates model input using gensim utilities """
def createInput(inputFile):
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

    # load corpus if it exists in path, otherwise create it
    if (os.path.exists(CORPUS_FILE_PATH)):
        corpus = corpora.MmCorpus(CORPUS_FILE_PATH)
    else:
        corpus = createCorpus(DATA_PATH)

    # load dictionary
    if (os.path.exists(DICTIONARY_FILE_PATH)):
        dictionary = loadDictionary(DICTIONARY_FILE_PATH)

    with open(inputFile, "wb") as output:
        LABEL_DICT = loadLabels(TWITTER_LABEL_PATH)
        trainOutputList = []
        testOutputList = []

        # dictionary.filter_n_most_frequent(10)
        keep_n_most_frequent(dictionary, 10)

        tfidf = get_complete_tfidf(corpus, dictionary)
        tfidf = [[t[1] for t in l] for l in tfidf]

        for idx, fname in enumerate(os.listdir(DATA_PATH)):
            # write list of word ids
            # document = getSequenceFromFile(os.path.join(DATA_PATH, fname))
            # document = [corpus.dictionary.token2id.get(w) for w in document]
            # line = (document, int(LABEL_DICT[fname.split('.')[0]]))

            if bytes(fname, encoding='utf-8') in trainSetList:
                line = (tfidf[idx], int(LABEL_DICT[fname.split('.')[0]]))
                trainOutputList.append(line)
            else:
                line = (tfidf[idx], int(LABEL_DICT[fname.split('.')[0]]))
                testOutputList.append(line)

        # write data pickle
        pickle.dump(trainOutputList, output)
        pickle.dump(testOutputList, output)


    # need to return X -> numpy of lists similar to imdb.load

""" Creates model input using keras utilities """
def createInputKeras(inputFile):
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    LABEL_DICT = loadLabels(TWITTER_LABEL_PATH)

    # load train filenames into list
    with open(TRAIN_SET_FILE_PATH, 'r') as f:
        for line in f:
            file_path = os.path.join(DATA_PATH, line.rstrip())
            X_train.append(getTextFromFile(file_path))
            y_train.append(int(LABEL_DICT[line.rstrip().split('.')[0]]))

    # load test filenames into list
    with open(TEST_SET_FILE_PATH, 'r') as f:
        for line in f:
            file_path = os.path.join(DATA_PATH, line.rstrip())
            X_test.append(getTextFromFile(file_path))
            y_test.append(int(LABEL_DICT[line.rstrip().split('.')[0]]))


    tokenizer_train = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer_test = Tokenizer(nb_words=MAX_NB_WORDS)

    tokenizer_train.fit_on_texts(X_train)
    tokenizer_test.fit_on_texts(X_test)

    sequences_train = tokenizer_train.texts_to_sequences(X_train)
    sequences_test = tokenizer_test.texts_to_sequences(X_test)

    word_index = tokenizer_train.word_index
    print('Found %s unique train tokens.' % len(word_index))

    X_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

    print('Shape of data_train tensor:', X_train.shape)

    with open(inputFile, "wb") as output:
        pickle.dump((X_train, y_train), output)
        pickle.dump((X_test, y_test), output)


""" Implements section 4.3 in the paper http://www.ijcai.org/Proceedings/16/Papers/537.pdf"""
def create_tfidfs(time_series):
    tfidf_series = []
    for interval in time_series:
        document = []
        for post in interval:
            document.append(post[0])
        word_count_vector = CountVectorizer().fit_transform(document)
        tfidf_transformer = TfidfTransformer()
        tfidf_vector = tfidf_transformer.fit_transform(word_count_vector)
        tfidf_series.append(tfidf_vector)
    return tfidf_series


""" Constructs variable length time series similar to section 4.2 in the paper http://www.ijcai.org/Proceedings/16/Papers/537.pdf"""
""" E: relevant posts (m_ij, t_ij)"""
""" N: reference length of RNN """
def variable_length_time_series(E, N):
    # Initialization
    L = (E[-1][1] - E[0][1]).total_seconds() // 60 # difference of time between last post and first post convert to minutes
    # L = (L.days*24*60) + (L.seconds // 60) #
    l = L // N # initial intervals
    l = int(l)
    k = 0
    U_hat_prev = []
    while True:
        k = k + 1
        U_k = Equipartition(E, N, l)
        U_hat_k = find_longest_time_span(U_k, l)
        if(len(U_hat_k) < N and len(U_hat_k) > len(U_hat_prev)):
            # shorten the intervals
            l = l // 2
            U_hat_prev = U_hat_k
        else:
            # generate output
            return U_hat_k

def Equipartition(E, N, l):
    U_k = []
    for i in range(N): # create N lists
        list = []
        for idx, rp in enumerate(E):
            left_most = datetime.timedelta(minutes = i*l) + E[0][1]
            right_most = datetime.timedelta(minutes = (i+1)*l) + E[0][1]
            if(left_most <= rp[1] and rp[1] < right_most):
                list.append(rp)
        U_k.append(list)
    return U_k

def find_longest_time_span(U_k, l):
    U_hat_k = []

    max_time_span_index = 0
    max_count = 0
    count = 0

    for idx, time_interval in enumerate(U_k):
        if time_interval:
            count = count + 1
        else:
            if count > max_count:
                max_count = count
                max_time_span_index = idx - count
            count = 0

    for i in range(max_time_span_index, max_time_span_index + max_count): # add longest time span to U_hat
        U_hat_k.append(U_k[i])

    return U_hat_k

def createCorpus(inputPath):
    corpus = RumorTextCorpus(inputPath)
    MmCorpus.serialize(CORPUS_FILE_PATH, corpus)
    corpus.dictionary.save(DICTIONARY_FILE_PATH)
    return corpus

def getSequenceFromFile(file_path):
    E = []
    for line in open(file_path):
        line = re.sub(' "source":(.[^,]+)",', '', line)  # remove json.loads corrupters
        jsonObject = json.loads(line)

        post = jsonObject['text']
        timestamp = jsonObject['created_at']
        timestamp = parser.parse(timestamp)

        e = (post, timestamp)
        E.append(e)
    E.sort(key=lambda x: x[1])
    return E

def getTextFromFile(file_path):
    W = ''
    for line in open(file_path):
        line = re.sub(' "source":(.[^,]+)",', '', line)  # remove json.loads corrupters
        jsonObject = json.loads(line)
        w = jsonObject['text']
        W = W + ' ' + w
    W = W[1:]
    return W

def loadDictionary(dictionaryPath):
    return corpora.Dictionary.load(dictionaryPath)

def loadCorpus(corpusPath):
    return MmCorpus(corpusPath)

class RumorTextCorpus(corpora.TextCorpus):

    def __init__(self, dirname):
        self.dirname = dirname
        super(RumorTextCorpus, self).__init__(dirname)


    # one line per document
    def get_texts(self):
        stoplist = set('for a of the and to in'.split()) # add http?
        for fname in os.listdir(self.dirname):
            W = []
            for line in open(os.path.join(self.dirname, fname)):
                line = re.sub(' "source":(.[^,]+)",', '', line)  # remove json.loads corrupters
                w = json.loads(line)

                # tokenize and remove common words
                w = utils.tokenize(w['text'], lowercase=True)
                w = [word for word in w if word not in stoplist]

                W.extend(w)
            yield W

if __name__ == "__main__": main()


