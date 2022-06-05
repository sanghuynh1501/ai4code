import random
import re
import string
from bisect import bisect

import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from wordcloud import STOPWORDS

from config import alphabet

nltk.download('wordnet')
nltk.download('omw-1.4')
stemmer = WordNetLemmatizer()
stopwords = set(STOPWORDS)


def shuffe_data(a, b):
    a = np.array(a)
    b = np.array(b)

    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)

    a = a[indices]
    b = b[indices]

    return a.tolist(), b.tolist()


def re_ranking_list(ids):
    sort_ids = ids.copy()
    sort_ids.sort()
    return [sort_ids.index(i) for i in ids]


def generate_data(df):
    data = []

    for id, df_tmp in tqdm(df.groupby('id')):
        source = df_tmp['cell_id'].to_list()
        rank = re_ranking_list(df_tmp['rank'].to_list())
        for i in range(len(source)):
            data.append([source[i], rank[i]])

    return data


def generate_data_sigmoid(df, mode='train'):
    data = []
    data_dict = {}
    for id, df_tmp in tqdm(df.groupby('id')):
        source = df_tmp['cell_id'].to_list()
        rank = re_ranking_list(df_tmp['rank'].to_list())
        if len(source) > 1:
            for i in range(len(source)):
                data.append([id, source[i], rank[i]])
            data_dict[id] = {
                'len': len(source),
                'source': source,
                'rank': rank
            }

    return data, data_dict


def generate_data_test(df):
    data = []

    for id, df_tmp in tqdm(df.groupby('id')):
        source = df_tmp['cell_id'].to_list()
        for i in range(len(source)):
            data.append(source[i])

    return data


def preprocess_text(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()
    # return document

    # Lemmatization
    tokens = document.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


def check_english(document):
    document = str(document)
    for char in document:
        if char not in list(string.ascii_lowercase) and char not in list(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and char != ' ':
            return 'other'
    return 'en'


def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):  # O(N)
        j = bisect(sorted_so_far, u)  # O(log N)
        inversions += i - j
        sorted_so_far.insert(j, u)  # O(N)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0  # total inversions in predicted ranks across all instances
    total_2max = 0  # maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        # rank predicted order in terms of ground truth
        ranks = [gt.index(x) for x in pred]
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


def map_result(ids, results):
    result_obj = {}
    for s, result in zip(ids, results):
        id = ''.join([alphabet[i] for i in s])
        result_obj[id] = result
    return result_obj


def sort_source(source, true_object):

    sorted_source = source.copy()

    sorted_source.sort(key=lambda i: true_object[i])

    return [sorted_source.index(s) for s in source]


def check_rank(df, true_object):
    ranks = []
    pred_ranks = []
    total = 0
    for _, df_tmp in tqdm(df.groupby('id')):

        source = df_tmp['cell_id'].to_list()
        rank = re_ranking_list(df_tmp['rank'].to_list())
        source, rank = shuffe_data(source, rank)

        pred_rank = sort_source(source, true_object)

        ranks.append(rank)
        pred_ranks.append(pred_rank)

        if total < 32:
            print(rank)
            print(pred_rank)
        total += 1

    return kendall_tau(ranks, pred_ranks)


def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )


def get_ranks(base, derived):
    return [base.index(d) for d in derived]


def adjust_lr(optimizer, epoch):
    if epoch < 1:
        lr = 5e-5
    elif epoch < 2:
        lr = 1e-3
    elif epoch < 5:
        lr = 1e-4
    else:
        lr = 1e-5

    for p in optimizer.param_groups:
        p['lr'] = lr
        
    return lr
