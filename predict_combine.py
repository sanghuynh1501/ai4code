import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from config import BS, CLASS_WEIGHTS, CODE_MARK_PATH, DATA_DIR, EPOCH, NW, RANK_COUNT, RANKS, accumulation_steps
from dataset import MarkdownDataset, MarkdownDatasetTest
from helper import get_features_val, kendall_tau
from torch.utils.data.dataloader import default_collate
from losses import FocalLoss

from model import MarkdownModel

device = 'cpu'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)

model = MarkdownModel()

df_orders = pd.read_csv(
    DATA_DIR / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

val_df = pd.read_csv('data_dump/val_df.csv')

with open('data_dump/dict_cellid_source.pkl', 'rb') as f:
    dict_cellid_source = pickle.load(f)
f.close()

unique_ids = pd.unique(val_df['id'])
ids = unique_ids[:1000]
val_df = val_df[val_df['id'].isin(ids)]
val_fts = get_features_val(val_df)

val_ds = MarkdownDatasetTest(
    dict_cellid_source, md_max_len=64, total_max_len=512, fts=val_fts)

val_loader = DataLoader(val_ds, batch_size=BS * 8, shuffle=False, num_workers=0,
                        pin_memory=False, drop_last=False)


def cal_kendall_tau(df, pred):
    index = 0
    df = df.sort_values('rank').reset_index(drop=True)
    df.loc[df['cell_type'] == 'code',
           'pred'] = df[df.cell_type == 'code']['rank']

    final_pred = {}

    for idx, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        code_sub_df_all = sub_df[sub_df.cell_type == 'code']

        for i in range(0, mark_sub_df_all.shape[0]):
            for j in range(0, code_sub_df_all.shape[0], RANK_COUNT):
                rank_index = RANKS[pred[index]]
                if rank_index >= 0 and rank_index < RANK_COUNT + 1:
                    code_sub_df = code_sub_df_all[j: j + RANK_COUNT]
                    if rank_index == 0:
                        cell_id = mark_sub_df_all.iloc[i]['cell_id']
                        final_pred[cell_id] = 0
                    else:
                        start_rank = 0
                        rank_index -= 1
                        if rank_index < len(code_sub_df):
                            start_rank = code_sub_df.iloc[rank_index]['rank']

                        cell_id = mark_sub_df_all.iloc[i]['cell_id']
                        final_pred[cell_id] = start_rank + 1
                index += 1

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


def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, (ids, mask, fts, code_lens, target) in enumerate(tbar):
            # with torch.cuda.amp.autocast():
            #     pred = model(ids.to(device), mask.to(device),
            #                  fts.to(device), code_lens.to(device))
            # pred = torch.argmax(pred, dim=1)
            # preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels)


label = validate(model, val_loader)
cal_kendall_tau(val_df, label)
