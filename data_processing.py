import csv
import pickle

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from config import DATA_DIR
from helper import (get_features_class, get_ranks, preprocess_code,
                    preprocess_text, read_notebook, write_json)

paths_train = list((DATA_DIR / 'train').glob('*.json'))
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

dict_cellid_source = dict(
    zip(df['cell_id'].values, df['source'].values))

NVALID = 0.1

splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
train_ind, val_ind = next(splitter.split(df, groups=df['ancestor_id']))

train_df = df.loc[train_ind].reset_index(drop=True)
val_df = df.loc[val_ind].reset_index(drop=True)

train_df.to_csv('data_dump/train_df.csv')
val_df.to_csv('data_dump/val_df.csv')

with open('data_dump/dict_cellid_source.pkl', 'wb') as handle:
    pickle.dump(dict_cellid_source, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

get_features_class('data_dump/json_train', train_df)
get_features_class('data_dump/json_val', val_df)