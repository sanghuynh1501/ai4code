import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BS, CODE_MARK_PATH, DATA_DIR, MAX_LEN, NW
from dataset import TestDataset
from helper import (generate_data_test, get_ranks, kendall_tau,
                    preprocess_code, preprocess_text, read_notebook)
from model import PairWiseModel

device = 'cpu'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)


def cal_distance(a, b):
    return model_pairse.get_score(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)).detach().cpu().numpy()


model_pairse = PairWiseModel().to(device)
model_pairse.load_state_dict(torch.load(CODE_MARK_PATH))
model_pairse.eval()

paths_train = list((DATA_DIR / 'train').glob('*.json'))[:10]
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

df.loc[df['cell_type'] == 'markdown', 'source'] = df[df['cell_type']
                                                     == 'markdown'].source.apply(preprocess_text)
df.loc[df['cell_type'] == 'code', 'source'] = df[df['cell_type']
                                                 == 'code'].source.apply(preprocess_code)

data = generate_data_test(df)

dict_cellid_source = dict(zip(df['cell_id'].values, df['source'].values))

val_data = TestDataset(data, dict_cellid_source, MAX_LEN)

val_loader = DataLoader(val_data, batch_size=BS * 2, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)

with torch.no_grad():
    cell_orders = None
    with tqdm(total=len(val_data)) as pbar:
        for ids, mask in val_loader:

            predicts = model_pairse.get_feature(ids.to(device), mask.to(
                device)).detach().cpu().numpy()

            if cell_orders is None:
                cell_orders = predicts
            else:
                cell_orders = np.concatenate([cell_orders, predicts], 0)

            pbar.update(len(ids))

# =======================================================================
feature_dict = dict(zip(df['cell_id'].values, cell_orders))
df[['id', 'cell_id', 'cell_type', 'rank']].to_csv('predict.csv')

for id, df_tmp in tqdm(df.groupby('id')):
    df_tmp_mark = df_tmp[df_tmp['cell_type'] == 'markdown']
    df_tmp_code = df_tmp[df_tmp['cell_type'] == 'code']

    df_tmp_mark_id = df_tmp_mark['cell_id'].values
    df_tmp_mark_rank = df_tmp_mark['rank'].values

    df_tmp_code_id = df_tmp_code['cell_id'].values
    df_tmp_code_rank = df_tmp_code['rank'].values

    print('================================================================')
    for i in range(len(df_tmp_mark)):
        for j in range(len(df_tmp_code)):
            if df_tmp_mark_rank[i] + 1 == df_tmp_code_rank[j]:
                print(df_tmp_mark_id[i], df_tmp_code_id[j], cal_distance(
                    feature_dict[df_tmp_mark_id[i]], feature_dict[df_tmp_code_id[j]]))
