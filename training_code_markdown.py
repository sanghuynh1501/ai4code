import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BS, CODE_MARK_PATH, DATA_DIR, MAX_LEN, NUM_TRAIN, NW
from dataset import PairWiseDataset, PairWiseRandomDataset
from helper import (adjust_lr, generate_triplet, generate_triplet_random,
                    get_ranks, read_notebook)
from model import PairWiseModel

device = 'cuda'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)

net = PairWiseModel().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters(
)), lr=3e-4, betas=(0.9, 0.999), eps=1e-08)


def train_step(ids, mask, labels):
    optimizer.zero_grad()

    outputs = net(ids, mask)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def test_step(ids, mask, labels):
    outputs = net(ids, mask)
    loss = criterion(outputs, labels)

    return loss.item()


def predict(ids, mask):
    predictions = net(ids, mask)

    return predictions


def convert_result(a):
    a = (a > 0.5).astype(np.int8)

    return a


paths_train = list((DATA_DIR / 'train').glob('*.json'))[:NUM_TRAIN]
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

NVALID = 0.1

splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)

train_ind, val_ind = next(splitter.split(df, groups=df['ancestor_id']))

train_df = df.loc[train_ind].reset_index(drop=True)
val_df = df.loc[val_ind].reset_index(drop=True)

data, dict_code = generate_triplet_random(train_df)
val_data = generate_triplet(val_df)

dict_cellid_source_train = dict(
    zip(train_df['cell_id'].values, train_df['source'].values))
dict_cellid_source_val = dict(
    zip(val_df['cell_id'].values, val_df['source'].values))

train_ds = PairWiseRandomDataset(
    data, dict_code, dict_cellid_source_train, MAX_LEN)
val_ds = PairWiseDataset(
    val_data, dict_cellid_source_val, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=NW,
                          pin_memory=False, drop_last=True)

val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)

EPOCHS = 5
max_test_tau = 0

for epoch in range(EPOCHS):

    train_loss = 0
    test_loss = 0

    train_trues, train_preds = [], []
    test_trues, test_preds, id_info_list = [], [], []

    train_tau = 0
    test_tau = 0

    train_accuracy = 0
    test_accuracy = 0

    total_train = 0
    total_test = 0

    total_true = 0

    net.train()
    lr = adjust_lr(optimizer, epoch)
    with tqdm(total=len(train_ds)) as pbar:
        for ids, mask, labels in train_loader:
            loss = train_step(ids.to(device), mask.to(
                device), labels.to(device))

            train_loss += loss * len(ids)
            total_train += len(ids)

            pbar.update(len(ids))

    net.eval()
    with torch.no_grad():
        with tqdm(total=len(val_ds)) as pbar:
            for ids, mask, labels in val_loader:
                loss = test_step(ids.to(device), mask.to(
                    device), labels.to(device))

                test_loss += loss * len(ids)
                total_test += len(ids)

                predicts = predict(ids.to(device), mask.to(
                    device)).detach().cpu().numpy().ravel()
                test_preds += predicts.tolist()

                total_true += np.sum(labels.cpu().numpy().ravel()
                                     == convert_result(predicts)).astype(np.int32)

                pbar.update(len(ids))

    test_accuracy = (total_true / total_test)
    torch.save(net.state_dict(), CODE_MARK_PATH)

    print(
        f'Epoch {epoch + 1}, \n'
        f'Loss: {train_loss / total_train}, '
        f'Test Loss: {test_loss / total_test}, '
        f'Test Accuracy: {test_accuracy * 100}, '
        f'Test Tau: {test_tau} '
    )
