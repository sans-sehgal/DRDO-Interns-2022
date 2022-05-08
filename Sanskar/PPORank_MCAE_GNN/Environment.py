import math
from Evaluation import DCG
from LoadData import *

class Dataset:
    def __init__(self, prev_shape):
        self.QUERY_TRAIN, _, self.QUERY_TEST, self.QUERY_DOC, self.QUERY_DOC_TRUTH, \
         self.DOC_REPR, self.MAX_DCG , self.prev_shape = transform_all(prev_shape)

    def getTrain(self):
        return self.QUERY_TRAIN

    # def getVali(self):
    #     return self.QUERY_VALI

    def getTest(self):
        return self.QUERY_TEST

    def getDocQuery(self):
        return self.QUERY_DOC

    def getTruth(self):
        return self.QUERY_DOC_TRUTH

    def getFeatures(self):
        return self.DOC_REPR

    def getIDCG(self):
        return self.MAX_DCG

    def getRelevance(self, Q, state):
        return [self.QUERY_DOC_TRUTH[Q][x] for x in state]

    def get_prev_shape(self):
        return self.prev_shape

def transform(dataset, num_features):
    train = import_dataset(dataset + "train.txt", n_features=num_features)
    test = import_dataset(dataset + "test.txt", n_features=num_features)
    vali = import_dataset(dataset + "vali.txt", n_features=num_features)
    all_data = {**train, **vali, **test}

    # List of queries
    QUERY_TRAIN = list(train)
    QUERY_VALI = list(vali)
    QUERY_TEST = list(test)

    # All queries and docs dictionary
    QUERY_DOC = {}
    DOC_REPR = {}
    QUERY_DOC_TRUTH = {}
    for query in list(all_data):
        QUERY_DOC[query] = list(all_data[query])
        QUERY_DOC_TRUTH[query] = {}
        for doc in list(all_data[query]):
            DOC_REPR[doc] = all_data[query][doc][0]
            QUERY_DOC_TRUTH[query][doc] = all_data[query][doc][1]

    # IDCG
    MAX_DCG = {}
    labels = {}
    for query in list(QUERY_DOC_TRUTH):
        MAX_DCG[query] = []
        labels[query] = []
        for doc in list(all_data[query]):
            labels[query].append(all_data[query][doc][1])
        for doc_pos in range(1, 11):
            if len(labels[query]) >= doc_pos:
                MAX_DCG[query].append(DCG(sorted(labels[query], reverse=True), doc_pos))
            else:
                MAX_DCG[query].append(0)

    return QUERY_TRAIN, QUERY_VALI, QUERY_TEST, QUERY_DOC, QUERY_DOC_TRUTH, DOC_REPR, MAX_DCG, prev_shape

# 
def transform_all(prev_shape):
    # train_data, test_data = import_all(dataset)
    train_data, test_data, prev_shape = generate_dataset(prev_shape)
    all_data = {**train_data, **test_data}

    # List of queries
    # List of queries
    QUERY_TRAIN = list(train_data)
    QUERY_TEST = list(test_data)

    # All queries and docs dictionary
    QUERY_DOC = {}
    DOC_REPR = {}
    QUERY_DOC_TRUTH = {}
    for query in list(all_data):
        QUERY_DOC[query] = list(all_data[query])
        QUERY_DOC_TRUTH[query] = {}
        for doc in list(all_data[query]):
            DOC_REPR[doc] = all_data[query][doc][0]
            QUERY_DOC_TRUTH[query][doc] = all_data[query][doc][1]

    # IDCG
    MAX_DCG = {}
    labels = {}
    for query in list(QUERY_DOC_TRUTH):
        MAX_DCG[query] = []
        labels[query] = []
        for doc in list(all_data[query]):
            labels[query].append(all_data[query][doc][1])
        for doc_pos in range(1, 11):
            if len(labels[query]) >= doc_pos:
                MAX_DCG[query].append(DCG(sorted(labels[query], reverse=True), doc_pos))
            else:
                MAX_DCG[query].append(0)

    return QUERY_TRAIN, None, QUERY_TEST, QUERY_DOC, QUERY_DOC_TRUTH, DOC_REPR, MAX_DCG, prev_shape


def get_reward(t, Y_at):
    """Calculates reward at time t given label Y_at """
    if t == 0:
        return (2 ** Y_at) - 1
    else:
        return ((2 ** Y_at) - 1) / float(math.log((t + 1), 2))


def update_state(t, Q, doc_action, state, QUERY_DOC_TRUTH):
    label = QUERY_DOC_TRUTH[Q][doc_action]
    reward = get_reward(t, label)
    X = []
    for items in state:
        if items != doc_action:
            X.append(items)

    # X = all documents
    # return state as documents
    return X, reward
