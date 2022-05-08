import numpy as np
from math import sqrt
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from Embeddings import *

def normalize(features):
    """L2 Normalization on input and returned"""
    norm = np.linalg.norm(features)
    if norm == 0:
        return features
    else:
        return features / norm


def mean(income):
    mn = 0.0
    for i in income:
        mn += i[0]
    return round(mn / len(income), 4)


def std_hand(income):
    income -= round(mean(income), 4)
    sump = 0.0
    for i in income:
        print(i[0], end=' ')
        sump += round(i[0] * i[0], 4)
        print(sump)
    return round(sqrt(sump / len(income)), 4)


def normalize_std(income):
    """Mean Normalization on input and returned"""
    print(f"Mean = {np.mean(income)} Std = {np.round(np.std(income))}")
    print(f"Mean hand = {mean(income)} STD hand = {std_hand(income)}")
    income -= mean(income)
    print(income)
    std_h = std_hand(income)
    if std_h == 0:
        return np.round(income, 4)
    return np.round((income / std_h), 4)


def normalize_minmax(features):
    """Min Max Scaling done on input and returned"""
    if np.max(features) - np.min(features) == 0:
        return features
    return (features - np.min(features)) / (np.max(features) - np.min(features))


def import_dataset(filename, n_features):
    """Imports file into a structure:{qid:{docid:(features,label),docid2:(features_2,label_2)},qid2:{docid:(),docid2:()}} """
    f = open(filename, "r")
    data = {}
    genid = 0
    for line in f.readlines():
        x = line.split(" ")
        label = int(x[0])
        qid = int(x[1].split(":")[1])
        features = []
        for i in range(n_features):
            features.append(float(x[2 + i].split(":")[1]))
        #         features = normalize(features)
        if "#docid" in x:
            docid = x[x.index("#docid") + 2]
        else:
            docid = str(genid + 1)
            genid += 1
        if qid not in data:
            data[qid] = {docid: (features, label)}
        else:
            data[qid][docid] = (features, label)
    return data


def import_all(filename):
    f = open(filename, 'rb')
    t = pickle.load(f)
    s = pd.Series(t)
    training_data, test_data = [i.to_dict() for i in train_test_split(s, train_size=0.7, random_state=4)]
    return training_data, test_data

# def get_dataset():
#     with open('/media/sanskar/Seagate Expansion Drive/DRDO/PPORank_MCAE/cora_nodes.p', 'rb') as f:
#         nodes = pickle.load(f)
#     with open('/media/sanskar/Seagate Expansion Drive/DRDO/PPORank_MCAE/cora_edges.p', 'rb') as f:
#         edges = pickle.load(f)
#     return nodes , edges
# prev_shape = 0
def generate_dataset(prev_shape):
    with open('/media/sanskar/Seagate Expansion Drive/DRDO/PPORank_MCAE/cora_nodes.p', 'rb') as f:
        nodes = pickle.load(f)
    with open('/media/sanskar/Seagate Expansion Drive/DRDO/PPORank_MCAE/cora_edges.p', 'rb') as f:
        edges = pickle.load(f)
    node_seq = set()
    edge_seq = set()
    query_seq = set()
    for node in nodes:
        node_seq.add(node[1])
        for edge in edges:
            if int(node[1]) == int(edge[0]) or int(node[1]) == int(edge[1]):
                node_seq.add(edge[0])
                node_seq.add(edge[1])
                edge_seq.add(edge)
        # array = get_embeddings(node_seq, edge_seq , 46)
        # print(array.shape)
        # if array.shape[0] > prev_shape:
        if len(node_seq) > prev_shape:
            array = get_embeddings(node_seq, edge_seq , 46)
            print(array.shape)
            for i in node_seq:
                for j in nodes:
                    if i == j[1]:
                        query_seq.add(j)
                        break
            final_dict = format_dataset(query_seq,node_seq,array)
            s = pd.Series(final_dict)
            training_data, test_data = [i.to_dict() for i in train_test_split(s, train_size=0.7, random_state=4)]
            prev_shape = array.shape[0]
            return training_data, test_data, prev_shape

# query_seq is -> query , docid, relevance score
# node_seq is -> just doc id's
# array is -> embeddings
def format_dataset(query_seq,node_seq,array):
    final_dict = {}
    temp_dict = {}
    embeddings_dict = {}
    for i,j in enumerate(node_seq):
        embeddings_dict[j] = array[i] 
    query_seq = sorted(query_seq)

    for i in range(len(query_seq)):
        if i == len(query_seq)-1:
            temp_dict[query_seq[i][1]] = (list(embeddings_dict[query_seq[i][1]]) , query_seq[i][2])
            final_dict[query_seq[i][0]] = temp_dict

        elif query_seq[i][0] == query_seq[i+1][0]:
            temp_dict[query_seq[i][1]] = (list(embeddings_dict[query_seq[i][1]]) , query_seq[i][2])
        else:
            temp_dict[query_seq[i][1]] = (list(embeddings_dict[query_seq[i][1]]) , query_seq[i][2])
            final_dict[query_seq[i][0]] = temp_dict
            temp_dict = {}

    return final_dict








