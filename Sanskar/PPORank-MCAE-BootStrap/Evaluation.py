import math
import numpy as np


def DCG(Y, k=None):
    """Calculates DCG at position k"""
    if k is None:
        k = len(Y)
    if k>len(Y):
        return 0
    val = 0
    for i in range(k):
        val += (((2 ** (abs(Y[i]))) - 1) / (math.log(i + 2, 2)))
    return val


def evaluate(E, X, k=None):
    """Calculates NDCG@k"""
    labels = [X[at[1]][1] for at in E]
    IDCG = DCG(sorted(labels, reverse=True), k)
    if IDCG == 0:
        # print("Can't compute NDCG as IDCG=0, returning DCG")
        return 0
    NDCG = DCG(labels, k) / IDCG
    return NDCG


def validate_individual(DOC_TRUTH, MAX_DCG, doc_action_list):
    label = []
    results = []
    for action in doc_action_list:
        label.append(DOC_TRUTH[action])

    for doc_pos in range(1, 11):
        if MAX_DCG[doc_pos - 1] > 0:
            results.append(DCG(label, doc_pos) / MAX_DCG[doc_pos - 1])
        else:
            results.append(0)

    return results


def calculate(dcg_results):  # dcg_results = {Q: [NDCG_scores]*10}
    final = [0] * 10
    counter = 0
    for query in dcg_results:
        # if len(dcg_results[query]) != 0:
        #     counter += 1
        for k in range(10):
            final[k] += dcg_results[query][k]
        # else:
        #     print(f"Query empty dcg {query}")
    # print(counter)
    return np.round(np.array(final) / len(dcg_results), 4)
    # return np.round(np.array(final), 4)
