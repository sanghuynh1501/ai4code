import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from config import BS, NW, SIGMOID_PATH
from dataset import SigMoidDataset
from helper import get_features_val, sigmoid_validate, sigmoid_validate_detail

from model import SigMoidModel

device = 'cuda'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)

model = SigMoidModel()
model.load_state_dict(torch.load(SIGMOID_PATH, map_location=device))
model = model.to(device)

val_df = pd.read_csv('data_dump/val_df.csv')
with open('data_dump/dict_cellid_source.pkl', 'rb') as f:
    dict_cellid_source = pickle.load(f)
f.close()

unique_ids = pd.unique(val_df['id'])
ids = unique_ids[:1000]
val_df = val_df[val_df['id'].isin(ids)]
val_df["pct_rank"] = val_df["rank"] / \
    val_df.groupby("id")["cell_id"].transform("count")


val_fts, _, _ = get_features_val(val_df, 'test')

val_ds = SigMoidDataset(dict_cellid_source, md_max_len=64,
                        total_max_len=512, fts=val_fts)

val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)

zero_accurancy, one_accurancy = sigmoid_validate_detail(model, val_loader, device)

print(zero_accurancy, one_accurancy)
