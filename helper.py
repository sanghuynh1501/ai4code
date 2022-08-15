import re
import sys
from bisect import bisect
import nltk
import numpy as np
import pandas as pd
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


def get_features_rank(df, dict_cellid_source, mode='train'):

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

            mark = mark_sub_df_all.iloc[i]['cell_id']
            rank = mark_sub_df_all.iloc[i]['rank']

            j = 0
            while j < code_sub_df_all.shape[0]:
                code_sub_df = code_sub_df_all[j: j + RANK_COUNT]

                codes = code_sub_df['cell_id'].to_list()
                ranks = code_sub_df['rank'].values
                total_code = code_sub_df.shape[0]

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
                else:
                    code_rank = RANKS[-1]

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
                    if total_code_len > RANK_COUNT and str(dict_cellid_source[mark]) != '':
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
                elif mode == 'train':
                    if str(dict_cellid_source[mark]) != '' and str(dict_cellid_source[mark]) != 'nan':
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

                j -= 1
                j += RANK_COUNT

    return np.array(features), np.array(labels), np.array(code_ranks)


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
            max_j = 0

            j = 0
            while j < code_sub_df_all.shape[0]:
                if relative[index] >= max_score:
                    max_score = relative[index]
                    max_index = index
                    max_j = j
                index += 1
                j -= 1
                j += RANK_COUNT

            cell_id = mark_sub_df_all.iloc[i]['cell_id']
            if RANKS[pred[max_index]] == 0:
                final_pred[cell_id] = -100
            else:
                rank_index = RANKS[pred[max_index]] - 1
                code_sub_df = code_sub_df_all[max_j: max_j + RANK_COUNT]
                if rank_index < code_sub_df.shape[0]:
                    final_pred[cell_id] = code_sub_df.iloc[rank_index]['rank'] + 1
                else:
                    # final_pred[cell_id] = code_sub_df.iloc[-1]['rank'] + 1
                    final_pred[cell_id] = mark_dict[cell_id] * sub_df.shape[0]

    for _, sub_df in tqdm(df.groupby('id')):
        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        cell_ids = mark_sub_df_all['cell_id'].to_list()

        for i in range(len(cell_ids) - 1):
            if final_pred[cell_ids[i]] == final_pred[cell_ids[i + 1]]:
                equal_list = [cell_ids[i]]
                for j in range(i + 1, len(cell_ids)):
                    if final_pred[cell_ids[j]] == final_pred[cell_ids[i]]:
                        equal_list.append(cell_ids[j])
                equal_list.sort(key=lambda id: mark_dict[id])
                for i, id in enumerate(equal_list):
                    final_pred[id] += i / 10

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

    return kendall_tau(df_orders.loc[y_dummy.index], y_dummy)


def validate_rank(model, val_loader, scriterion, ccriterion, device):
    model.eval()
    total_loss = 0
    total_step = 0

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    relatives = []

    with torch.no_grad():
        for idx, (ids, mask, fts, code_len, target, realtive) in enumerate(tbar):
            with torch.cuda.amp.autocast():
                spred, cped = model(ids.to(device), mask.to(device),
                                    fts.to(device), code_len.to(device))

                loss = (scriterion(spred, realtive.to(device)) +
                        ccriterion(cped, torch.squeeze(target).to(device))) / 2

            total_loss += loss.detach().cpu().item()
            total_step += 1

            pred = torch.argmax(cped, dim=1)
            relative = torch.sigmoid(spred)

            preds.append(pred.detach().cpu().numpy().ravel())
            relatives.append(relative.detach().cpu().numpy().ravel())

    preds, relatives = np.concatenate(preds), np.concatenate(relatives)
    avg_loss = np.round(total_loss / total_step, 4)

    return preds, relatives, avg_loss


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
