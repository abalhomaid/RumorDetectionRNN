"""TEST METHODS, YOU CAN IGNORE THIS FILE"""

# returns N by M matrix
# where N is number of documents
# and M is number of words in corpus
def get_complete_tfidf(corpus, dictionary):
    tfidf = models.TfidfModel(corpus)
    num_unique = len(dictionary)
    # num_unique = 12
    input = tfidf[corpus]
    output = []
    for idx, doc in enumerate(input):
        wordIDs = [t[0] for t in doc]
        word_tfidf = [t[1] for t in doc]
        added = []
        for i in list(range(num_unique)):
        #     if i not in wordIDs:
        #         added.append((i, 0))
        # doc.extend(added)
        # doc.sort(key=lambda tup: tup[0])
        # output.append(doc)
            if i not in wordIDs:
                added.append((i, 0))
            else:
                added.append(i, word_tfidf[i])

    return output

def createTfidf(corpus, tfidfSavePath = None):
    tfidf = models.TfidfModel(corpus)

    if tfidfSavePath is not None:
        tfidf.save(tfidfSavePath)

    return tfidf


def keep_n_most_frequent(d, remove_n):
    # determine which tokens to keep
    most_frequent_ids = (v for v in six.itervalues(d.token2id))
    most_frequent_ids = sorted(most_frequent_ids, key=d.dfs.get, reverse=True)
    most_frequent_ids = most_frequent_ids[:remove_n]
    # do the actual filtering, then rebuild dictionary to remove gaps in ids
    most_frequent_words = [(d[id], d.dfs.get(id, 0)) for id in most_frequent_ids]
    d.filter_tokens(good_ids=most_frequent_ids)

# creates dictionary for all words in the dataset
def createDictionary():
    sentences = MySentences(DATA_PATH)
    dictionary = Dictionary(sentences)
    dictionary.save(DICTIONARY_FILE_PATH)
    return dictionary

def testGensim():
    hello = Dictionary(["máma mele maso".split(), "ema má máma".split()])
    print('what')

    corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
    [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
    [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
    [(0, 1.0), (4, 2.0), (7, 1.0)],
    [(3, 1.0), (5, 1.0), (6, 1.0)],
    [(9, 1.0)],
    [(9, 1.0), (10, 1.0)],
    [(9, 1.0), (10, 1.0), (11, 1.0)],
    [(8, 1.0), (10, 1.0), (11, 1.0)]]

    output = get_complete_tfidf(corpus)

    vec = [(0, 1), (4, 1)]
    # print(tfidf[vec])

def loadTfidf(tfidfPath):
    return models.TfidfModel.load(tfidfPath)

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