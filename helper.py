import re
import sys
from bisect import bisect

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from wordcloud import STOPWORDS

from config import RANK_COUNT, RANKS

nltk.download('wordnet')
nltk.download('omw-1.4')
stemmer = WordNetLemmatizer()
stopwords = set(STOPWORDS)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


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


def get_features_class(df):

    features = {}
    df = df.sort_values('rank').reset_index(drop=True)

    for idx, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        code_sub_df_all = sub_df[sub_df.cell_type == 'code']

        min_rank = code_sub_df_all['rank'].min()
        total_md = mark_sub_df_all.shape[0]

        for j in range(0, code_sub_df_all.shape[0], RANK_COUNT):
            code_sub_df = code_sub_df_all[j: j + RANK_COUNT]

            codes = code_sub_df['cell_id'].to_list()
            ranks = code_sub_df['rank'].to_list()
            total_code = code_sub_df.shape[0]

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


def check_code_by_rank(rank, full_codes_ranks):
    return rank in full_codes_ranks


def get_features_val(df, mode='train'):

    features = []
    labels = []
    relatives = []
    df = df.sort_values('rank').reset_index(drop=True)

    for idx, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        code_sub_df_all = sub_df[sub_df.cell_type == 'code']
        total_code_len = len(code_sub_df_all)

        min_rank = code_sub_df_all['rank'].min()
        full_code_rank = code_sub_df_all['rank'].to_list()
        total_md = mark_sub_df_all.shape[0]

        for i in range(0, mark_sub_df_all.shape[0]):
            for j in range(0, code_sub_df_all.shape[0], RANK_COUNT):
                code_sub_df = code_sub_df_all[j: j + RANK_COUNT]

                codes = code_sub_df['cell_id'].to_list()
                ranks = code_sub_df['rank'].values
                total_code = code_sub_df.shape[0]

                mark = mark_sub_df_all.iloc[i]['cell_id']
                rank = mark_sub_df_all.iloc[i]['rank']
                relative = 1

                sub_ranks = []
                if rank < min_rank:
                    rank = 0
                else:
                    sub_ranks = rank - ranks
                    sub_ranks_positive = sub_ranks[sub_ranks > 0]
                    if len(sub_ranks_positive) == 0:
                        rank = -1
                        relative = 0
                    else:
                        sub_ranks[sub_ranks < 0] = 100000
                        rank = np.argmin(sub_ranks) + 1
                        if rank == RANK_COUNT:
                            for r in range(ranks[-1] + 1, mark_sub_df_all.iloc[i]['rank'], 1):
                                if check_code_by_rank(r, full_code_rank):
                                    rank = -1
                                    relative = 0

                if mode == 'train':
                    if rank != -1:
                        feature = {
                            'total_code': int(total_code),
                            'total_md': int(total_md),
                            'codes': codes,
                            'mark': mark,
                            'rank': int(rank),
                            'pct_rank': mark_sub_df_all.iloc[i]['pct_rank'],
                            'relative': relative,
                            'total_code_len': total_code_len
                        }

                        features.append(feature)
                        labels.append(RANKS.index(int(rank)))

                        relatives.append(relative)
                elif mode == 'sigmoid':
                    if total_code_len > RANK_COUNT:
                        feature = {
                            'total_code': int(total_code),
                            'total_md': int(total_md),
                            'codes': codes,
                            'mark': mark,
                            'rank': int(rank) if rank != -1 else 0,
                            'pct_rank': mark_sub_df_all.iloc[i]['pct_rank'],
                            'relative': relative,
                            'total_code_len': total_code_len
                        }

                        features.append(feature)
                        relatives.append(relative)
                else:
                    feature = {
                        'total_code': int(total_code),
                        'total_md': int(total_md),
                        'codes': codes,
                        'mark': mark,
                        'rank': int(rank) if rank != -1 else 0,
                        'pct_rank': mark_sub_df_all.iloc[i]['pct_rank'],
                        'relative': relative,
                        'total_code_len': total_code_len
                    }

                    features.append(feature)
                    relatives.append(relative)

    return features, labels, relatives


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


def cal_kendall_tau(df, pred, relative, df_orders):
    index = 0
    df = df.sort_values('rank').reset_index(drop=True)
    df.loc[df['cell_type'] == 'code',
           'pred'] = df[df.cell_type == 'code']['pct_rank']

    final_pred = {}

    for _, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        code_sub_df_all = sub_df[sub_df.cell_type == 'code']

        for i in range(0, mark_sub_df_all.shape[0]):
            max_score = 0
            max_index = 0
            max_j = 0
            for j in range(0, code_sub_df_all.shape[0], RANK_COUNT):
                if relative[index] >= max_score:
                    max_score = relative[index]
                    max_index = index
                    max_j = j
                index += 1

            # rank_index = RANKS[pred[max_index]]
            # code_sub_df = code_sub_df_all[max_j: max_j + RANK_COUNT]
            # if rank_index == 0:
            #     cell_id = mark_sub_df_all.iloc[i]['cell_id']
            #     final_pred[cell_id] = 0
            # else:
            #     start_rank = 0
            #     rank_index -= 1
            #     if rank_index < len(code_sub_df):
            #         start_rank = code_sub_df.iloc[rank_index]['rank']

            cell_id = mark_sub_df_all.iloc[i]['cell_id']
            final_pred[cell_id] = pred[max_index]

    pred = []
    cell_ids = []
    for cell_id in final_pred.keys():
        cell_ids.append(cell_id)
        pred.append(final_pred[cell_id])

    df_markdown_pred = pd.DataFrame(list(zip(cell_ids, pred)), columns=[
                                    'cell_id', 'markdown_pred'])
    df = df.merge(df_markdown_pred, on=['cell_id'], how='outer')

    df.loc[df['cell_type'] == 'markdown',
           'pred'] = df.loc[df['cell_type'] == 'markdown']['markdown_pred']

    df[['id', 'cell_id', 'cell_type', 'rank', 'pred']].to_csv('predict.csv')
    y_dummy = df.sort_values("pred").groupby('id')['cell_id'].apply(list)
    print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))


def sigmoid_validate(model, val_loader, device):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    total = 0
    total_true = 0
    relatives = []

    with torch.no_grad():
        for idx, (ids, mask, fts, _, code_lens, _, target, total_code_lens) in enumerate(tbar):
            with torch.cuda.amp.autocast():
                pred = model(ids.to(device), mask.to(device),
                             fts.to(device), code_lens.to(device))

            code_lens = (total_code_lens.detach().cpu().numpy().ravel()
                         <= RANK_COUNT)
            pred = pred.detach().cpu().numpy().ravel()
            pred = (sigmoid(pred) >= 0.5)
            pred = (pred | code_lens).astype(np.int8)
            # pred = pred + code_lens
            pred = np.clip(pred, 0, 1)
            relatives += pred.tolist()
            target = target.detach().cpu().numpy().ravel()
            total += len(target)
            total_true += np.sum((pred == target).astype(np.int8))

    return total_true / total, relatives


def validate(model, val_loader, device):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    targets = []
    code_lens = []

    with torch.no_grad():
        for idx, (ids, mask, fts, loss_mask, code_len, target, _, total_code_len) in enumerate(tbar):
            with torch.cuda.amp.autocast():
                pred = model(ids.to(device), mask.to(device),
                             fts.to(device), code_len.to(device), loss_mask.to(device))
            # pred = torch.argmax(pred, dim=1)
            preds.append(pred.detach().cpu().numpy().ravel())
            targets.append(target.detach().cpu().numpy().ravel())
            code_lens.append(total_code_len.detach().cpu().numpy().ravel())

    preds, targets, code_lens = np.concatenate(
        preds), np.concatenate(targets), np.concatenate(code_lens)
    acc_index = np.where(code_lens <= RANK_COUNT)
    return preds, targets, 0
    # accuracy_score(targets[acc_index], preds[acc_index])
