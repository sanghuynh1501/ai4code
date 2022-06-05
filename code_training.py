import random
import string
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
# from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BS, MAX_LEN, NVALID, NW
from dataset import DatasetTest, MarkdownDataset
from helper import (adjust_lr, check_rank, generate_data_sigmoid,
                    generate_data_test, map_result)
from model import MarkdownModel

device = 'cuda'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)


net = MarkdownModel().to(device)
# net.load_state_dict(torch.load('code_model.pth'))
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4,
                       betas=(0.9, 0.999), eps=1e-08)


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


extra_df = pd.read_csv('csv/extra_dataset.csv')
extra_df = extra_df[extra_df['cell_type'] == 'code']
extra_df = extra_df[extra_df['source'].notnull()]
random_strings = [''.join(random.choice(string.ascii_lowercase)
                          for i in range(8)) for _ in range(len(extra_df))]
extra_df['cell_id'] = random_strings

df = pd.read_csv('csv/code_dataset.csv')
df = df[df['source'].notnull()]
df = pd.concat([df, extra_df]).reset_index(drop=True)

df_size = df.groupby(['id']).size().to_frame()
df_size.columns = ['size']
df_size.reset_index(inplace=True)

df_len = df.groupby('id')['source'].agg(
    lambda x: x.str.split().apply(len).max()).to_frame()
df_len.columns = ['len']
df_len.reset_index(inplace=True)

df_size = df_size.merge(df_len, on='id', how='left')
df_size = df_size[(df_size['size'] <= 100) & (df_size['len'] <= 128)]

concat_df_size = []
for i in range(100):
    sub_df = df_size[df_size['size'] == i + 1].reset_index(drop=True)[:100]
    concat_df_size.append(sub_df)

df_size = pd.concat(concat_df_size).reset_index(drop=True)
df = df_size.merge(df, on='id', how='left')

splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
train_ind, val_ind = next(splitter.split(df, groups=df['id']))

train_df = df.loc[train_ind].reset_index(drop=True)
val_df = df.loc[val_ind].reset_index(drop=True)

data, data_dic = generate_data_sigmoid(train_df, 'train')
val_data, val_data_dic = generate_data_sigmoid(val_df, 'test')
test_data = generate_data_test(val_df)

dict_cellid_source_train = dict(
    zip(train_df['cell_id'].values, train_df['source'].values))
dict_cellid_source_val = dict(
    zip(val_df['cell_id'].values, val_df['source'].values))

train_ds = MarkdownDataset(
    data, data_dic, dict_cellid_source_train, MAX_LEN)
val_ds = MarkdownDataset(
    val_data, val_data_dic, dict_cellid_source_val, MAX_LEN)
test_ds = DatasetTest(
    test_data, dict_cellid_source_val, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=NW,
                          pin_memory=False, drop_last=True)

val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)

test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False,
                         num_workers=NW, pin_memory=False, drop_last=False)

EPOCHS = 200
max_test_tau = 0

for epoch in range(EPOCHS):

    train_loss = 0
    test_loss = 0

    train_trues, train_preds = [], []
    test_trues, test_preds, id_info_list = [], [], []

    train_tau = 0
    test_tau = 0
    test_tau_full = 0
    train_accuracy = 0
    test_accuracy = 0
    total_true = 0

    net.train()
    lr = adjust_lr(optimizer, epoch)

    with tqdm(total=len(train_ds)) as pbar:
        for ids, mask, labels in train_loader:
            loss = train_step(ids.to(device), mask.to(
                device), labels.to(device))
            train_loss += loss * BS

            pbar.update(len(ids))

    net.eval()
    with torch.no_grad():
        with tqdm(total=len(val_ds)) as pbar:
            for ids, mask, labels in val_loader:
                loss = test_step(ids.to(device), mask.to(
                    device), labels.to(device))
                test_loss += loss * BS

                predicts = predict(ids.to(device), mask.to(
                    device)).detach().cpu().numpy().ravel()

                test_trues += labels.cpu().numpy().tolist()

                total_true += np.sum(labels.cpu().numpy().ravel()
                                     == convert_result(predicts)).astype(np.int32)

                pbar.update(len(ids))

    net.eval()
    with torch.no_grad():
        with tqdm(total=len(test_ds)) as pbar:
            for ids, mask, id_infoes in test_loader:

                predicts = net.score(ids.to(device), mask.to(
                    device)).detach().cpu().numpy().ravel()

                id_info_list += id_infoes.cpu().numpy().tolist()
                test_preds += predicts.tolist()

                pbar.update(len(ids))

    result_obj = map_result(id_info_list, test_preds)
    test_tau = check_rank(val_df, result_obj)
    test_accuracy = (total_true / len(test_trues))

    torch.save(net.state_dict(), 'code_model.pth')

    print(
        f'Epoch {epoch + 1}, \n'
        f'Loss: {train_loss / len(train_ds)}, '
        f'Accuracy: {train_accuracy * 100}, '
        f'Tau: {train_tau}, \n'
        f'Test Loss: {test_loss / len(val_ds)}, '
        f'Test Accuracy: {test_accuracy * 100}, '
        f'Test Tau: {test_tau} '
    )
