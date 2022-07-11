import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve
import torch
from torch.utils.data import DataLoader
from config import BS, DATA_DIR, NW, SIGMOID_PATH
from dataset import SigMoidDataset
from helper import get_features_new, get_features_rank, get_features_val, sigmoid_validate, sigmoid_validate_detail, test_cal_kendall_tau_rank

from model import SigMoidModel

# device = 'cuda'
# torch.cuda.empty_cache()
# np.random.seed(0)
# torch.manual_seed(0)

# model = SigMoidModel()
# model.load_state_dict(torch.load(SIGMOID_PATH, map_location=device))
# model = model.to(device)

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

val_df.drop('rank', axis=1, inplace=True)
val_df["rank"] = val_df.groupby(["id", "cell_type"]).cumcount()
val_df = val_df.sort_values('rank').reset_index(drop=True)
val_df["pct_rank"] = val_df["rank"] / \
    val_df.groupby("id")["cell_id"].transform("count")


val_fts, _ = get_features_rank(val_df, 'test')
# classes = []
# for ft in val_fts:
#     classes.append(ft['code_rank'])

# rank_dict = {}
# cell_ids = val_df['cell_id'].to_list()
# ranks = val_df['rank'].to_list()

# for id, rank in zip(cell_ids, ranks):
#     rank_dict[id] = rank

# total_error = 0

# for ft in val_fts:
#     index = ft['code_rank']
#     if index == 0:
#         rank = 0
#     else:
#         rank = rank_dict[ft['codes'][index - 1]] + 1
#     mark = ft['mark']

#     if abs(rank - rank_dict[mark]) > 1:
#         total_error += 1
#         print(ft['id'], mark, rank, rank_dict[mark])

# print(total_error, len(val_fts))
test_cal_kendall_tau_rank(val_df, val_fts, df_orders)
# print(np.unique(classes))
# counts, bins = np.histogram(classes)
# plt.hist(bins[:-1], bins, weights=counts)
# plt.show()

# val_ds = SigMoidDataset(dict_cellid_source, md_max_len=64,
#                         total_max_len=512, fts=val_fts)

# val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=NW,
#                         pin_memory=False, drop_last=False)

# acc, true, false, relative, targets, preds = sigmoid_validate(
#     model, val_loader, device)

# print(acc, true, false)

# precision, recall, thresholds = precision_recall_curve(targets, preds)
# # convert to f score
# fscore = (2 * precision * recall) / (precision + recall)
# # locate the index of the largest f score
# ix = np.argmax(fscore)
# print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

# acc, true, false, relative, targets, preds = sigmoid_validate(
#     model, val_loader, device, thresholds[ix])

# print('precision_recall: ', acc, true, false)
