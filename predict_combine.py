import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (BS, CODE_MARK_PATH, CODE_PATH, DATA_DIR, MARK_PATH,
                    MAX_LEN, NW)
from dataset import TestDataset
from helper import generate_data_test, get_ranks, kendall_tau, read_notebook
from model import PairWiseModel, ScoreModel

device = 'cuda'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)


def cal_distance(a, b):
    d = a - b
    return 1/(1 + np.exp(-d))


def get_code(mar, codes, pairs_hash):
    max_code = ''
    max_score = 0
    for code in codes:
        score = cal_distance(pairs_hash[mar], pairs_hash[code])
        if score > max_score:
            max_score = score
            max_code = code
    return max_code


model_code = ScoreModel().to(device)
model_code.load_state_dict(torch.load(CODE_PATH))
model_code.eval()

model_markdown = ScoreModel().to(device)
model_markdown.load_state_dict(torch.load(MARK_PATH))
model_markdown.eval()

model_pairse = PairWiseModel().to(device)
model_pairse.load_state_dict(torch.load(CODE_MARK_PATH))
model_pairse.eval()

paths_train = list((DATA_DIR / 'train').glob('*.json'))[10000:11000]
notebooks_train = [
    read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
]

df = (
    pd.concat(notebooks_train)
    .set_index('id', append=True)
    .swaplevel()
    .sort_index(level='id', sort_remaining=False)
)

df_orders = pd.read_csv(
    DATA_DIR / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

df_orders_ = df_orders.to_frame().join(
    df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
    how='right',
)

ranks = {}
for id_, cell_order, cell_id in df_orders_.itertuples():
    ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

df_ranks = (
    pd.DataFrame
    .from_dict(ranks, orient='index')
    .rename_axis('id')
    .apply(pd.Series.explode)
    .set_index('cell_id', append=True)
)

df_ancestors = pd.read_csv(DATA_DIR / 'train_ancestors.csv', index_col='id')

df = df.reset_index().merge(
    df_ranks, on=['id', 'cell_id']).merge(df_ancestors, on=['id'])

df_code = df[df['cell_type'] == 'code'].reset_index(drop=True)
df_mark = df[df['cell_type'] == 'markdown'].reset_index(drop=True)

data_code = generate_data_test(df_code)
data_mark = generate_data_test(df_mark)
data_pair = generate_data_test(df)

dict_cellid_source = dict(zip(df['cell_id'].values, df['source'].values))

val_code = TestDataset(data_code, dict_cellid_source, MAX_LEN)
val_mark = TestDataset(data_mark, dict_cellid_source, MAX_LEN)
val_pair = TestDataset(data_pair, dict_cellid_source, MAX_LEN)

loader_code = DataLoader(val_code, batch_size=BS * 2, shuffle=False, num_workers=NW,
                         pin_memory=False, drop_last=False)

loader_mark = DataLoader(val_mark, batch_size=BS * 2, shuffle=False, num_workers=NW,
                         pin_memory=False, drop_last=False)

loader_pair = DataLoader(val_pair, batch_size=BS * 2, shuffle=False, num_workers=NW,
                         pin_memory=False, drop_last=False)

with torch.no_grad():
    code_orders = []
    with tqdm(total=len(val_code)) as pbar:
        for ids, mask in loader_code:

            predicts = model_code(ids.to(device), mask.to(
                device)).detach().cpu().numpy().ravel()
            code_orders += predicts.tolist()

            pbar.update(len(ids))

    mark_orders = []
    with tqdm(total=len(val_mark)) as pbar:
        for ids, mask in loader_mark:

            predicts = model_markdown(ids.to(device), mask.to(
                device)).detach().cpu().numpy().ravel()
            mark_orders += predicts.tolist()

            pbar.update(len(ids))

    pair_orders = []
    with tqdm(total=len(val_pair)) as pbar:
        for ids, mask in loader_pair:

            predicts = model_pairse.score(ids.to(device), mask.to(
                device)).detach().cpu().numpy().ravel()
            pair_orders += predicts.tolist()

            pbar.update(len(ids))

pairs_hash = {}
for i in range(len(data_pair)):
    pairs_hash[data_pair[i]] = pair_orders[i]

# =======================================================================
df_code['pred'] = code_orders
y_dummy = df_code.sort_values('pred').groupby('id')['cell_id'].apply(list)
test_tau = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
print('test tau code ', test_tau)

df_code = y_dummy.to_frame()
df_code.reset_index(inplace=True)
codes = df_code['cell_id'].tolist()

# =======================================================================
df_mark['pred'] = mark_orders
y_dummy = df_mark.sort_values('pred').groupby('id')['cell_id'].apply(list)
test_tau = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
print('test tau mark ', test_tau)

df_mark = y_dummy.to_frame()
df_mark.reset_index(inplace=True)
marks = df_mark['cell_id'].tolist()

results = []
for mark, code in zip(marks, codes):
    result = []
    for ma in mark:
        cd = get_code(ma, code, pairs_hash)
        result.append(ma)
        result.append(cd)

    results.append(result)

test_tau = kendall_tau(df_orders.loc[y_dummy.index], results)
print('test tau ', test_tau)
