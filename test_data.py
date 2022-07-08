import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from config import (BS, CODE_MARK_PATH, DATA_DIR, EPOCH, MARK_PATH, MD_MAX_LEN, NW, SIGMOID_PATH, TOTAL_MAX_LEN,
                    accumulation_steps)
from dataset import MarkdownDataset, MarkdownOnlyDataset, SigMoidDataset
from helper import cal_kendall_tau, get_features_mark, get_features_new, get_features_val, markdown_validate, sigmoid_validate, validate
from model import MarkdownModel, MarkdownOnlyModel, SigMoidModel

# device = 'cuda'
# torch.cuda.empty_cache()
# np.random.seed(0)
# torch.manual_seed(0)

# model = MarkdownModel()
# # model.load_state_dict(torch.load(CODE_MARK_PATH))
# model = model.cuda()

# model_sigmoid = SigMoidModel().to(device)
# model_sigmoid.load_state_dict(torch.load(SIGMOID_PATH))
# model_sigmoid = model_sigmoid.cuda()

# model_mark_only = MarkdownOnlyModel()
# model_mark_only.load_state_dict(torch.load(MARK_PATH))
# model_mark_only = model_mark_only.cuda()

df_orders = pd.read_csv(DATA_DIR / 'train_orders.csv')

train_df = pd.read_csv('data_dump/train_df.csv')
val_df = pd.read_csv('data_dump/val_df.csv')
with open('data_dump/dict_cellid_source.pkl', 'rb') as f:
    dict_cellid_source = pickle.load(f)
f.close()

unique_ids = pd.unique(train_df['id'])
ids = unique_ids[:100]
train_df = train_df[train_df['id'].isin(ids)]
train_df["pct_rank"] = train_df["rank"] / \
    train_df.groupby("id")["cell_id"].transform("count")

unique_ids = pd.unique(val_df['id'])
ids = unique_ids[:100]
val_df = val_df[val_df['id'].isin(ids)]
val_df["pct_rank"] = val_df["rank"] / \
    val_df.groupby("id")["cell_id"].transform("count")

train_fts, _ = get_features_new(train_df, 'sigmoid')
val_fts, _ = get_features_new(val_df, 'test')
val_fts_only = get_features_mark(val_df, 'test')

isStart = 0 
isLast = 0
isNormal = 0
for ft in val_fts:
    if ft['isStart']:
        isStart += 1
    if ft['isLast']:
        isLast += 1
    if not ft['isStart'] and not ft['isLast']:
        isNormal += 1

print(isStart / len(val_fts), isLast / len(val_fts), isNormal / len(val_fts))
# ids = df_orders['id'].to_list()
# cells = df_orders['cell_order'].to_list()

# order_dict = {}
# for id, cell in zip(ids, cells):
#     order_dict[id] = cell.split(' ')

# for ft in val_fts:
#     orders = order_dict[ft['id']]
#     codes = ft['codes']
#     relative = ft['relative']
#     mark = ft['mark']
#     code_start = orders.index(codes[0])
#     code_end = orders.index(codes[-1])
#     mark_index = orders.index(mark)
#     rank = mark_index / len(orders)
#     if code_start > code_end:
#         print('error index')
#     if relative == 1:
#         if mark not in orders[code_start:code_end]:
#             print(mark, codes[0], codes[-1])
#             print('error not in')
#     else:
#         if mark in orders[code_start:code_end]:
#             print('error not in')


# for ft in val_fts:
#     orders = order_dict[ft['id']]
#     codes = ft['codes']
#     code_start = orders.index(codes[0])
#     code_end = orders.index(codes[-1])
#     if code_start > code_end:
#         print('error index')
#     if relative == 1:
#         if mark not in orders[code_start:code_end]:
#             print('error not in')
#     else:
#         if mark in orders[code_start:code_end]:
#             print('error not in')
