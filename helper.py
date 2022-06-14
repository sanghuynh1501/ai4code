import re
from bisect import bisect

import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from wordcloud import STOPWORDS

from config import RANK_COUNT

nltk.download('wordnet')
nltk.download('omw-1.4')
stemmer = WordNetLemmatizer()
stopwords = set(STOPWORDS)


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
    return str(cell).replace('\\n', '\n')[:200]


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


def sample_cells(cells, n):
    cells = [preprocess_code(cell) for cell in cells]
    if n >= len(cells):
        return [cell[:200] for cell in cells]
    else:
        results = []
        step = len(cells) / n
        idx = 0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in results
        if cells[-1] not in results:
            results[-1] = cells[-1]
        return results


def get_features(df):
    features = dict()
    df = df.sort_values('rank').reset_index(drop=True)
    for idx, sub_df in tqdm(df.groupby('id')):
        features[idx] = dict()
        total_md = sub_df[sub_df.cell_type == 'markdown'].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == 'code']
        total_code = code_sub_df.shape[0]
        codes = sample_cells(code_sub_df.source.values, 20)
        features[idx]['total_code'] = total_code
        features[idx]['total_md'] = total_md
        features[idx]['codes'] = codes
    return features


def get_features_class(df):

    features = {}
    df = df.sort_values('rank').reset_index(drop=True)

    for idx, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        code_sub_df_all = sub_df[sub_df.cell_type == 'code']

        min_rank = code_sub_df_all['rank'].min()
        total_md = mark_sub_df_all.shape[0]
        total_code = mark_sub_df_all.shape[0]

        for j in range(0, code_sub_df_all.shape[0], RANK_COUNT):
            code_sub_df = code_sub_df_all[j: j + RANK_COUNT]
            codes = code_sub_df['cell_id'].to_list()
            ranks = code_sub_df['rank'].to_list()

            feature = {
                'total_code': int(total_code),
                'total_md': int(total_md),
                'codes': codes,
                'ranks': ranks,
                'min_rank': min_rank
            }

            if idx not in features:
                features[idx] = [feature]
            else:
                features[idx].append(feature)

    return features


def get_features_val(df):

    features = []
    df = df.sort_values('rank').reset_index(drop=True)

    for idx, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        code_sub_df_all = sub_df[sub_df.cell_type == 'code']

        min_rank = code_sub_df_all['rank'].min()
        total_md = mark_sub_df_all.shape[0]
        total_code = mark_sub_df_all.shape[0]

        for i in range(0, mark_sub_df_all.shape[0]):
            for j in range(0, code_sub_df_all.shape[0], RANK_COUNT):
                code_sub_df = code_sub_df_all[j: j + RANK_COUNT]
                codes = code_sub_df['cell_id'].to_list()
                ranks = code_sub_df['rank'].values
                mark = mark_sub_df_all.iloc[i]['cell_id']
                rank = mark_sub_df_all.iloc[i]['rank']

                if rank < min_rank:
                    rank = 0
                else:
                    sub_ranks = rank - ranks
                    sub_ranks = sub_ranks[sub_ranks > 0]
                    if len(sub_ranks) == 0:
                        rank = -1
                    else:
                        rank = np.argmin(sub_ranks)
                        if rank > 20:
                            rank = 21

                feature = {
                    'total_code': int(total_code),
                    'total_md': int(total_md),
                    'codes': codes,
                    'mark': mark,
                    'rank': int(rank)
                }

                features.append(feature)

    return features


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