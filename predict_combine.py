import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BS, CODE_MARK_PATH, DATA_DIR, MAX_LEN, NW
from dataset import MarkdownDataset, TestDataset
from helper import (generate_data_test, generate_triplet, get_ranks, kendall_tau,
                    preprocess_code, preprocess_text, read_data, read_notebook)
from model import MarkdownModel, PairWiseModel

device = 'cuda'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)


model = MarkdownModel().to(device)
model.load_state_dict(torch.load(CODE_MARK_PATH))
model.eval()

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

df.source = df.source.apply(preprocess_text)

data = generate_triplet(df, 'test')

print('=============================================')
print('data ', len(data))
print('=============================================')

dict_cellid_source = dict(zip(df['cell_id'].values, df['source'].values))

val_data = MarkdownDataset(data, dict_cellid_source, MAX_LEN)

val_loader = DataLoader(val_data, batch_size=BS * 2, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)

with torch.no_grad():
    preds = []
    with tqdm(total=len(val_data)) as pbar:
        for data in val_loader:

            inputs, target = read_data(data)

            pred = model(inputs[0], inputs[1])

            preds += pred.detach().cpu().numpy().ravel().tolist()

            pbar.update(len(inputs[0]))

for id, df_tmp in tqdm(df.groupby('id')):

    df_tmp_markdown = df_tmp[df_tmp['cell_type'] == 'markdown']
    df_tmp_code = df_tmp[df_tmp['cell_type'] == 'code']

    idx = 0
    for cell_id, rank in df_tmp_markdown[['cell_id', 'rank']].values:
        for cid, crank in df_tmp_code[['cell_id', 'rank']].values:
            if crank != rank + 1:
                print(cell_id, cid, rank, crank, preds[idx])
            idx += 1
