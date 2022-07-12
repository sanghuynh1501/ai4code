import re
import sys
from bisect import bisect

import nltk
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from wordcloud import STOPWORDS

from config import LABELS, RANK_COUNT, RANKS

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


def check_code_by_rank(rank, full_codes_ranks):
    return rank in full_codes_ranks


def get_features_rank(df, mode='train'):

    features = []
    labels = []
    code_ranks = []
    df = df.sort_values('rank').reset_index(drop=True)

    for id, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        code_sub_df_all = sub_df[sub_df.cell_type == 'code']
        total_code_len = len(code_sub_df_all)
        total_md = mark_sub_df_all.shape[0]

        for i in range(0, mark_sub_df_all.shape[0]):
            for j in range(0, code_sub_df_all.shape[0], RANK_COUNT):
                code_sub_df = code_sub_df_all[j: j + RANK_COUNT]

                codes = code_sub_df['cell_id'].to_list()
                ranks = code_sub_df['rank'].values
                total_code = code_sub_df.shape[0]

                mark = mark_sub_df_all.iloc[i]['cell_id']
                rank = mark_sub_df_all.iloc[i]['rank']

                min_rank = 0 if j == 0 else ranks[0]
                max_rank = ranks[-1]

                relative = 1

                if total_code_len - j <= RANK_COUNT and rank > min_rank:
                    relative = 1
                else:
                    if rank < min_rank or rank > max_rank:
                        relative = 0

                code_rank = 0
                if relative == 1:
                    if j == 0 and rank < ranks[0]:
                        code_rank = 0
                    else:
                        sub_ranks = rank - ranks
                        sub_ranks[sub_ranks < 0] = 100000
                        code_rank = np.argmin(sub_ranks) + 1

                if len(ranks) < RANK_COUNT:
                    ranks = np.concatenate(
                        [ranks, np.ones(RANK_COUNT - len(ranks),) * ranks[-1]], 0)

                if mode == 'classification':
                    if relative == 1:
                        feature = {
                            'id': id,
                            'total_code': int(total_code),
                            'total_md': int(total_md),
                            'codes': codes,
                            'ranks': ranks,
                            'code_rank': code_rank,
                            'mark': mark,
                            'pct_rank': mark_sub_df_all.iloc[i]['pct_rank'],
                            'relative': relative,
                            'total_code_len': total_code_len
                        }
                        features.append(feature)
                elif mode == 'sigmoid':
                    if total_code_len > RANK_COUNT:
                        feature = {
                            'total_code': int(total_code),
                            'total_md': int(total_md),
                            'codes': codes,
                            'ranks': ranks,
                            'code_rank': code_rank,
                            'mark': mark,
                            'pct_rank': mark_sub_df_all.iloc[i]['pct_rank'],
                            'relative': relative,
                            'total_code_len': total_code_len
                        }
                        features.append(feature)
                else:
                    feature = {
                        'total_code': int(total_code),
                        'total_md': int(total_md),
                        'codes': codes,
                        'ranks': ranks,
                        'code_rank': code_rank,
                        'mark': mark,
                        'pct_rank': mark_sub_df_all.iloc[i]['pct_rank'],
                        'relative': relative,
                        'total_code_len': total_code_len
                    }
                    features.append(feature)
                labels.append(relative)
                code_ranks.append(code_rank)

    return np.array(features), np.array(code_ranks)


def get_features_mark(df, mode='train'):

    features = []
    df = df.sort_values('rank').reset_index(drop=True)

    for _, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']

        for i in range(0, mark_sub_df_all.shape[0]):
            mark = mark_sub_df_all.iloc[i]['cell_id']
            pct_rank = mark_sub_df_all.iloc[i]['pct_rank']

            feature = {
                'mark': mark,
                'pct_rank': pct_rank
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


def cal_kendall_tau_rank(df, pred, mark_dict, relative, df_orders):
    index = 0
    df = df.sort_values('rank').reset_index(drop=True)
    df.loc[df['cell_type'] == 'code',
           'pred'] = df[df.cell_type == 'code']['rank']

    final_pred = {}

    for _, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        code_sub_df_all = sub_df[sub_df.cell_type == 'code']

        for i in range(0, mark_sub_df_all.shape[0]):
            max_score = 0
            max_index = 0
            one_count = 0
            max_j = 0
            for j in range(0, code_sub_df_all.shape[0], RANK_COUNT):
                if relative[index] >= max_score:
                    max_score = relative[index]
                    max_index = index
                    one_count += 1
                    max_j = j
                index += 1
            cell_id = mark_sub_df_all.iloc[i]['cell_id']
            if max_score == 0 or one_count > 1:
                final_pred[cell_id] = mark_dict[cell_id] * sub_df.shape[0]
            else:
                if RANKS[pred[max_index]] == 0:
                    final_pred[cell_id] = 0
                else:
                    rank_index = RANKS[pred[max_index]] - 1
                    code_sub_df = code_sub_df_all[max_j: max_j + RANK_COUNT]
                    if rank_index < code_sub_df.shape[0]:
                        final_pred[cell_id] = code_sub_df.iloc[rank_index]['rank'] + 1
                    else:
                        final_pred[cell_id] = mark_dict[cell_id] * \
                            sub_df.shape[0]

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


def validate_sigmoid(model, val_loader, device, threshold=0.5):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    total = 0
    zero_total = 0
    one_total = 0

    total_true = 0
    total_zero_true = 0
    total_one_true = 0
    relatives = []

    preds = []
    targets = []

    with torch.no_grad():
        for idx, (ids, mask, fts, _, code_lens, _, target, total_code_lens) in enumerate(tbar):
            with torch.cuda.amp.autocast():
                pred = model(ids.to(device), mask.to(device),
                             fts.to(device), code_lens.to(device))

            code_lens = (total_code_lens.detach().cpu().numpy().ravel()
                         <= RANK_COUNT).astype(np.int8)
            code_len_indexs = np.nonzero(code_lens == 1)[0]

            pred = torch.sigmoid(pred)
            pred = pred.detach().cpu().numpy().ravel()
            pred[code_len_indexs] = 1.0
            preds += pred.tolist()

            pred = (pred >= threshold).astype(np.int8)
            # pred = (pred | code_lens).astype(np.int8)
            # pred = pred + code_lens
            # pred = np.clip(pred, 0, 1)
            relatives += pred.tolist()

            target = target.detach().cpu().numpy().ravel()
            targets += target.tolist()

            zero_indexes = np.nonzero(target == 0)[0]
            one_indexes = np.nonzero(target == 1)[0]

            zero_target = target[zero_indexes]
            one_target = target[one_indexes]

            zero_pred = pred[zero_indexes]
            one_pred = pred[one_indexes]

            zero_total += len(zero_target)
            one_total += len(one_target)
            total += len(target)

            total_zero_true += np.sum((zero_pred ==
                                       zero_target).astype(np.int8))
            total_one_true += np.sum((one_pred == one_target).astype(np.int8))
            total_true += np.sum((pred == target).astype(np.int8))

    return total_true / total, total_zero_true / zero_total, total_one_true / one_total, relatives, targets, preds


def validate_rank(model, val_loader, device):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    targets = []
    code_lens = []

    with torch.no_grad():
        for idx, (ids, mask, fts, _, code_len, target, _, total_code_len) in enumerate(tbar):
            with torch.cuda.amp.autocast():
                pred = model(ids.to(device), mask.to(device),
                             fts.to(device), code_len.to(device))
            pred = torch.argmax(pred, dim=1)
            preds.append(pred.detach().cpu().numpy().ravel())
            targets.append(target.detach().cpu().numpy().ravel())
            code_lens.append(total_code_len.detach().cpu().numpy().ravel())

    preds, targets, code_lens = np.concatenate(
        preds), np.concatenate(targets), np.concatenate(code_lens)

    return preds, targets, 0


def validate_markdown(model, val_loader, device):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    mark_ids = []
    mark_hash = {}

    with torch.no_grad():
        for idx, (ids, mask, _, id) in enumerate(tbar):
            with torch.cuda.amp.autocast():
                pred = model(ids.to(device), mask.to(device))
            preds += pred.detach().cpu().numpy().ravel().tolist()
            mark_ids += [label_to_id(i) for i in id]

    for mark, score in zip(mark_ids, preds):
        mark_hash[mark] = score

    return mark_hash


def id_to_label(ids):
    return [LABELS.index(s) for s in ids]


def label_to_id(labels):
    return ''.join([LABELS[i] for i in labels])


def cal_kendall_tau_inference(df, mark_dict, final_pred, df_orders):
    df.loc[df['cell_type'] == 'code',
           'pred'] = df[df.cell_type == 'code']['rank']

    marks = df.loc[df['cell_type'] == 'markdown']['cell_id'].to_list()
    for mark in marks:
        if mark not in final_pred:
            final_pred[mark] = mark_dict[mark]

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


def validate_rank_inference(model, val_loader, device):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    targets = []
    mark_ids = []
    mark_dict = {}
    rank_list = []

    with torch.no_grad():
        for _, (ids, mask, fts, code_len, target, cell_id, ranks) in enumerate(tbar):
            ranks = ranks.detach().cpu().numpy().tolist()
            with torch.cuda.amp.autocast():
                pred = model(ids.to(device), mask.to(device),
                             fts.to(device), code_len.to(device))
            pred = torch.argmax(pred, dim=1)
            preds.append(pred.detach().cpu().numpy().ravel())
            targets.append(target.detach().cpu().numpy().ravel())
            mark_ids += [label_to_id(i) for i in cell_id]
            rank_list += ranks

    preds, targets = np.concatenate(preds), np.concatenate(targets)

    for (id, pred, rank) in zip(mark_ids, preds, rank_list):
        if pred == 0:
            mark_dict[id] = pred
        else:
            mark_dict[id] = rank[pred - 1] + 1

    return preds, targets, accuracy_score(targets, preds), mark_dict
