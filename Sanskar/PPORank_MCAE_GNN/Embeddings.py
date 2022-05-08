import argparse
import pickle

import networkx as nx
from node2vec import Node2Vec

import numpy as np 
import pandas as pd

def get_embeddings(idx , s , dims):
    graph = nx.DiGraph()
    graph.add_nodes_from(idx)
    graph.add_edges_from(s)

    assert graph.number_of_nodes() == len(idx)
    assert graph.number_of_edges() == len(s)

    n2v = Node2Vec(graph, dimensions=dims, walk_length=20, num_walks=10, workers=1)
    n2v_model = n2v.fit(window=2, min_count=1, epochs=10, seed=123)
    return n2v_model.wv.vectors