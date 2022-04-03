import pandas as pd 
import numpy as np 
import csv
import gzip
import pickle
from sklearn.model_selection import train_test_split
import json
import os
import re
import gzip
import csv
import multiprocessing
import scipy
import numpy as np
import time

df = pd.read_csv('MSTop100.csv')
df.drop(columns = ['Unnamed: 0'] , inplace=True)
f = open('data_feature_vectors', 'rb')
dict = pickle.load(f)


def final_dataset(df , dict):
    print('generating final dataset ... ')
    dict_query = {}
    qid = df['qid'][0]
    
    temp_dict = {}
    for i in df.index:
            qid_prev = qid
            qid = df.loc[i , 'qid']
            # print(qid_prev , qid)
            if qid_prev != qid:
                dict_query[qid_prev] = temp_dict
                temp_dict = {}
            docid = df.loc[i , 'docid']
            data_vector = list(dict[docid])
            score = df.loc[i,'score']
            temp_dict[docid] = (data_vector , score)
            if i % 1000000 == 0:
                print(i)

    print('storing data as pickle file ... ')
    filehandler = open('complete_msmarco' , 'wb')
    pickle.dump(dict_query, filehandler)
    filehandler.close()

if __name__ == '__main__':
    final_dataset(df , dict)