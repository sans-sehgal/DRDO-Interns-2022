import math
from Evaluation import DCG
from LoadData import import_dataset, import_all
import numpy as np

class Dataset:
    def __init__(self, dataset):
        self.QUERY_TRAIN, _, self.QUERY_TEST, self.QUERY_DOC, self.QUERY_DOC_TRUTH, \
            self.DOC_REPR, self.MAX_DCG, self.QUERY_VEC = transform_all(dataset)

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

    def getQVEC(self,Q):
        return self.QUERY_VEC[Q]

    def updateQVEC(self,Q,modified):
        self.QUERY_VEC[Q]=modified
        #print(self.QUERY_VEC)

    def updateRelevance(self,Q,doc,SS):
        #self.QUERY_DOC_TRUTH[Q][doc] = self.QUERY_DOC_TRUTH[Q][doc]+5
        '''
        if self.QUERY_DOC_TRUTH[Q][doc] == 0:
            self.QUERY_DOC_TRUTH[Q][doc] = 2
        else:
            self.QUERY_DOC_TRUTH[Q][doc]+=2.0/(self.QUERY_DOC_TRUTH[Q][doc])
        '''
        self.QUERY_DOC_TRUTH[Q][doc]+=SS


    def updateIDCG(self,Q):
        self.MAX_DCG = RecalculateIDCG(Q,self.QUERY_DOC_TRUTH,self.MAX_DCG)


def transform_all(dataset):
    train_data, test_data = import_all(dataset)
    #print(train_data)
    all_data = {**train_data, **test_data}

    # List of queries
    # List of queries
    QUERY_TRAIN = list(train_data)
    QUERY_TEST = list(test_data)

    # All queries and docs dictionary
    QUERY_DOC = {}
    DOC_REPR = {}
    QUERY_DOC_TRUTH = {}
    QUERY_VEC = {}
    for query in list(all_data):
        QUERY_DOC[query] = list(all_data[query])
        QUERY_DOC_TRUTH[query] = {}
        QUERY_VEC[query] = np.array([float(0)]*46)
        count=0
        for doc in list(all_data[query]):
            count+=1
            DOC_REPR[doc] = all_data[query][doc][0]
            QUERY_VEC[query]+=np.array(DOC_REPR[doc])
            QUERY_DOC_TRUTH[query][doc] = all_data[query][doc][1]

        QUERY_VEC[query] = QUERY_VEC[query]/count
        #print(QUERY_VEC[query])
        #print('----------')

    #print(QUERY_DOC)
    #example of key value pair 14284: ['GX000-18-157110', 'GX001-88-338181', 'GX013-60-212804', 'GX024-62-1009296', 'GX028-25-924244', 'GX128-54-372176', 'GX244-62-329559', 'GX251-70-782965']
    #print(DOC_REPR)
    # example data 'GX244-62-329559': [0.011952, 0.0, 0.0, 0.0, 0.007843, 0.0, 0.0, 0.0, 0.0, 0.0, 0.010807, 0.0, 0.0, 0.0, 0.007542, 0.007097, 0.416667, 0.428571, 0.5, 0.007219, 0.638364, 0.621127, 0.481351, 0.840693, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.62744, 0.606546, 0.457585, 0.827606, 0.333333, 0.5, 0.0, 0.093714, 0.25, 0.0]
    #print(QUERY_DOC_TRUTH)
    #14284: {'GX000-18-157110': 0, 'GX001-88-338181': 0, 'GX013-60-212804': 0, 'GX024-62-1009296': 0, 'GX028-25-924244': 1, 'GX128-54-372176': 0, 'GX244-62-329559': 0, 'GX251-70-782965': 2}

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

    #print(QUERY_VEC)
    #print(MAX_DCG)
    #14284: [3.0, 3.6309297535714573, 3.6309297535714573, 3.6309297535714573, 3.6309297535714573, 3.6309297535714573, 3.6309297535714573, 3.6309297535714573, 0, 0]

    return QUERY_TRAIN, None, QUERY_TEST, QUERY_DOC, QUERY_DOC_TRUTH, DOC_REPR, MAX_DCG, QUERY_VEC

def RecalculateIDCG(query,QUERY_DOC_TRUTH,MAX_DCG):

    #IDCG
    #MAX_DCG = {}
    labels = {}
    MAX_DCG[query] = []
    labels[query] = []
    for doc in list(QUERY_DOC_TRUTH[query]):
        labels[query].append(QUERY_DOC_TRUTH[query][doc])
    #print(labels)
    for doc_pos in range(1, 11):
        if len(labels[query]) >= doc_pos:
            MAX_DCG[query].append(DCG(sorted(labels[query], reverse=True), doc_pos))
        else:
            MAX_DCG[query].append(0)


    return MAX_DCG 




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
