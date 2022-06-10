import re
import string
from bisect import bisect

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from wordcloud import STOPWORDS

nltk.download('wordnet')
nltk.download('omw-1.4')
stemmer = WordNetLemmatizer()
stopwords = set(STOPWORDS)


def generate_data(df):
    data = []

    for id, df_tmp in tqdm(df.groupby('id')):
        source = df_tmp['cell_id'].to_list()
        rank = df_tmp['pct_rank'].to_list()
        for i in range(len(source)):
            data.append([source[i], rank[i]])

    return data


def generate_data_test(df):
    data = []

    for id, df_tmp in tqdm(df.groupby('id')):
        source = df_tmp['cell_id'].to_list()
        for i in range(len(source)):
            data.append(source[i])

    return data


def generate_mark_code_dict(df):
    triplets = {}

    for id, df_tmp in tqdm(df.groupby('id')):
        df_tmp_markdown = df_tmp[df_tmp['cell_type'] == 'markdown']

        df_tmp_code = df_tmp[df_tmp['cell_type'] == 'code']
        df_tmp_code_rank = df_tmp_code['rank'].values
        df_tmp_code_cell_id = df_tmp_code['cell_id'].values

        for cell_id, rank in df_tmp_markdown[['cell_id', 'rank']].values:
            labels = np.array([(r == (rank+1))
                              for r in df_tmp_code_rank]).astype('int')

            for cid, label in zip(df_tmp_code_cell_id, labels):
                if label == 1:
                    triplets[cell_id] = cid

    return triplets


def generate_triplet_random(df):
    triplets = []
    dict_code = {}

    for id, df_tmp in tqdm(df.groupby('id')):
        df_tmp_markdown = df_tmp[df_tmp['cell_type'] == 'markdown']

        df_tmp_code = df_tmp[df_tmp['cell_type'] == 'code']
        df_tmp_code_cell_id = df_tmp_code['cell_id'].values
        df_tmp_code_rank = df_tmp_code['rank'].values

        if len(df_tmp_code) > 0 and len(df_tmp_markdown) > 0:
            for cell_id, rank in df_tmp_markdown[['cell_id', 'rank']].values:
                triplets.append([id, cell_id, rank])

            dict_code[id] = {
                'len': len(df_tmp_code_cell_id),
                'codes': df_tmp_code_cell_id,
                'ranks': df_tmp_code_rank
            }

    return triplets, dict_code


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


def preprocess_code(cell):
    return str(cell).replace('\\n', '\n')


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


def get_token(mark_id, code_id, dict_cellid_source, max_len, tokenizer):
    ids = []
    masks = []

    for id in [mark_id, code_id]:
        txt = dict_cellid_source[id]

        inputs = tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        id = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        id = torch.unsqueeze(id, 0)
        mask = torch.unsqueeze(mask, 0)

        ids.append(id)
        masks.append(mask)

    return torch.unsqueeze(torch.cat(ids, 0), 0), torch.unsqueeze(torch.cat(masks, 0), 0)
