from distutils import text_file
import re
import sys
from bisect import bisect
from matplotlib import pyplot as plt

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from wordcloud import STOPWORDS

from config import LABELS, RANK_COUNT, RANKS, SIGMOID_RANK_COUNT

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


def get_cosine_features(mark, codes, dict_cellid_source, vectorizer):
    codes = [dict_cellid_source[code] for code in codes]
    mark = [dict_cellid_source[mark]]

    codes = vectorizer.transform(codes).toarray()
    mark = vectorizer.transform(mark).toarray()[0]

    similarity_scores = codes.dot(
        mark) / (np.linalg.norm(codes, axis=1) * np.linalg.norm(mark))
    if len(similarity_scores) < SIGMOID_RANK_COUNT:
        similarity_scores = np.concatenate(
            [similarity_scores, np.ones(SIGMOID_RANK_COUNT - len(similarity_scores),)], 0)

    return similarity_scores


def get_features_sigmoid_text_all(df, dict_cellid_source, mode='train'):

    features = []
    labels = []
    df = df.sort_values('rank').reset_index(drop=True)

    for _, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        code_sub_df_all = sub_df[sub_df.cell_type == 'code']
        total_code_len = len(code_sub_df_all)

        for i in range(0, mark_sub_df_all.shape[0]):
            for j in range(0, code_sub_df_all.shape[0], SIGMOID_RANK_COUNT):
                code_sub_df = code_sub_df_all[j: j + SIGMOID_RANK_COUNT]

                codes = code_sub_df['cell_id'].to_list()
                ranks = code_sub_df['rank'].values

                mark = mark_sub_df_all.iloc[i]['cell_id']
                rank = mark_sub_df_all.iloc[i]['rank']

                min_rank = 0 if j == 0 else ranks[0]
                max_rank = ranks[-1]

                relative = 1
                if total_code_len - j <= SIGMOID_RANK_COUNT:
                    relative = 1
                else:
                    if rank < min_rank or rank > max_rank:
                        relative = 0

                if mode == 'train':
                    if total_code_len > RANK_COUNT:
                        text_all = ''
                        for code in codes:
                            text_all += (dict_cellid_source[code] + ' ')
                        text_all += dict_cellid_source[mark]
                        features.append(text_all)
                        labels.append(relative)
                else:
                    text_all = ''
                    for code in codes:
                        text_all += (dict_cellid_source[code] + ' ')
                    text_all += dict_cellid_source[mark]
                    features.append(text_all)
                    labels.append(relative)

    return features, np.array(labels)


def get_features_new(df, mode='train'):

    features = []
    df = df.sort_values('rank').reset_index(drop=True)

    for _, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        code_sub_df_all = sub_df[sub_df.cell_type == 'code']
        total_code_len = len(code_sub_df_all)
        total_md = mark_sub_df_all.shape[0]

        for i in range(0, mark_sub_df_all.shape[0]):
            for j in range(0, code_sub_df_all.shape[0], SIGMOID_RANK_COUNT):
                code_sub_df = code_sub_df_all[j: j + SIGMOID_RANK_COUNT]

                codes = code_sub_df['cell_id'].to_list()
                ranks = code_sub_df['rank'].values
                total_code = code_sub_df.shape[0]

                mark = mark_sub_df_all.iloc[i]['cell_id']
                rank = mark_sub_df_all.iloc[i]['rank']

                min_rank = 0 if j == 0 else ranks[0]
                max_rank = ranks[-1]

                relative = 1
                if total_code_len - j <= SIGMOID_RANK_COUNT:
                    relative = 1
                else:
                    if rank < min_rank or rank > max_rank:
                        relative = 0

                if mode == 'classification':
                    if relative == 1:
                        feature = {
                            'total_code': int(total_code),
                            'total_md': int(total_md),
                            'codes': codes,
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
                        'mark': mark,
                        'pct_rank': mark_sub_df_all.iloc[i]['pct_rank'],
                        'relative': relative,
                        'total_code_len': total_code_len
                    }
                    features.append(feature)

    return np.array(features)


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


def cal_kendall_tau(df, pred, mark_dict, relative, df_orders):
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
            one_count = 0
            for _ in range(0, code_sub_df_all.shape[0], RANK_COUNT):
                if relative[index] >= max_score:
                    max_score = relative[index]
                    max_index = index
                    one_count += 1
                index += 1
            cell_id = mark_sub_df_all.iloc[i]['cell_id']
            if max_score == 0 or one_count > 1:
                final_pred[cell_id] = mark_dict[cell_id]
            else:
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


def sigmoid_validate_detail(model, val_loader, device):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    zero_total = 0
    one_total = 0
    total_zero_true = 0
    total_one_true = 0

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
            pred = np.clip(pred, 0, 1)
            target = target.detach().cpu().numpy().ravel()

            zero_indexes = np.where(np.any(target == 0))
            one_indexes = np.where(np.any(target == 1))

            zero_target = target[zero_indexes]
            one_target = target[one_indexes]

            zero_pred = pred[zero_indexes]
            one_pred = pred[one_indexes]

            zero_total += len(zero_target)
            one_total += len(one_target)

            total_zero_true += np.sum((zero_pred ==
                                      zero_target).astype(np.int8))
            total_one_true += np.sum((one_pred == one_target).astype(np.int8))

    return total_zero_true / zero_total, total_one_true / one_total


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

    return preds, targets, 0


def markdown_validate(model, val_loader, device):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    mark_ids = []
    mark_hash = {}

    with torch.no_grad():
        for idx, (ids, mask, target, id) in enumerate(tbar):
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


def plot_cosin_distance(X, y):
    zero_index = np.nonzero(y == 0)[0]
    one_index = np.nonzero(y == 1)[0]

    x1 = X[zero_index, 0]
    y1 = X[zero_index, 1]

    # dataset2
    x2 = X[one_index, 0]
    y2 = X[one_index, 1]

    plt.scatter(x1, y1, c="green",
                linewidths=2,
                marker="s",
                edgecolor="green",
                s=len(x1))

    plt.scatter(x2, y2, c="red",
                linewidths=2,
                marker="s",
                edgecolor="red",
                s=len(x2))

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
