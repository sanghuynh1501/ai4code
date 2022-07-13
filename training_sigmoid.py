import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from config import (BS, CODE_MARK_RANK_PATH, DATA_DIR,
                    EPOCH, MARK_PATH, NW, SIGMOID_PATH, accumulation_steps)
from dataset import MarkdownOnlyDataset, SigMoidDataset
from helper import cal_kendall_tau_rank, get_features_mark, get_features_rank, validate_markdown, validate_sigmoid, validate_rank
from loss import BinaryFocalLossWithLogits
from model import MarkdownOnlyModel, MarkdownRankModel, SigMoidModel

device = 'cuda'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)

model = SigMoidModel()
model.load_state_dict(torch.load(SIGMOID_PATH))
model = model.cuda()

model_mark = MarkdownRankModel()
model_mark.load_state_dict(torch.load(CODE_MARK_RANK_PATH))
model_mark = model_mark.cuda()

model_mark_only = MarkdownOnlyModel()
model_mark_only.load_state_dict(torch.load(MARK_PATH))
model_mark_only = model_mark_only.cuda()

df_orders = pd.read_csv(
    DATA_DIR / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

train_df = pd.read_csv('data_dump/train_df.csv')
val_df = pd.read_csv('data_dump/val_df.csv')
with open('data_dump/dict_cellid_source.pkl', 'rb') as f:
    dict_cellid_source = pickle.load(f)
f.close()

# unique_ids = pd.unique(train_df['id'])
# ids = unique_ids[:100]
# train_df = train_df[train_df['id'].isin(ids)]
train_df["pct_rank"] = train_df["rank"] / \
    train_df.groupby("id")["cell_id"].transform("count")

unique_ids = pd.unique(val_df['id'])
ids = unique_ids[:100]
val_df = val_df[val_df['id'].isin(ids)]
val_df["pct_rank"] = val_df["rank"] / \
    val_df.groupby("id")["cell_id"].transform("count")

train_fts, all_labels, _ = get_features_rank(train_df, 'sigmoid')
val_fts, _, _ = get_features_rank(val_df, 'test')
val_fts_only = get_features_mark(val_df, 'test')

all_labels.sort()
label_dict = {}
for label in all_labels:
    if label not in label_dict:
        label_dict[label] = 0
    else:
        label_dict[label] += 1

class_weights = [label_dict[key] / len(all_labels)
                 for key in label_dict.keys()]

train_ds = SigMoidDataset(
    dict_cellid_source, md_max_len=64, total_max_len=512, fts=train_fts)
val_ds = SigMoidDataset(dict_cellid_source, md_max_len=64,
                        total_max_len=512, fts=val_fts)
val_ds_only = MarkdownOnlyDataset(val_fts_only, dict_cellid_source, 128)

train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=NW,
                          pin_memory=False, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BS * 8, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)
val_loader_only = DataLoader(val_ds_only, batch_size=BS, shuffle=False, num_workers=NW,
                             pin_memory=False, drop_last=False)

score, _, _ = validate_rank(model_mark, val_loader, device)
mark_dict = validate_markdown(model_mark_only, val_loader_only, device)


def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(
        epochs * len(train_loader) / accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = BinaryFocalLossWithLogits(
        alpha=class_weights[-1], gamma=2.0, reduction='mean')
    # criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    for e in range(epochs):
        model.train()
        total_loss = 0
        total_step = 0
        tbar = tqdm(train_loader, file=sys.stdout)

        for idx, (ids, mask, fts, _, code_lens, _, target, _) in enumerate(tbar):
            with torch.cuda.amp.autocast():
                pred = model(ids.to(device), mask.to(device),
                             fts.to(device), code_lens.to(device))
                loss = criterion(pred, target.to(device))
            scaler.scale(loss).backward()
            if idx % accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.detach().cpu().item()
            total_step += 1

            avg_loss = np.round(total_loss / total_step, 4)

            tbar.set_description(
                f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")

            if (idx + 1) % 10000 == 0 or idx == len(tbar) - 1:
                acc, true, false, relative, _, _ = validate_sigmoid(
                    model, val_loader, device, 0.397705)
                print('accurancy ', acc, true, false)
                cal_kendall_tau_rank(
                    val_df, score, mark_dict, relative, df_orders)
                torch.save(model.state_dict(), SIGMOID_PATH)
                model.train()


train(model, train_loader, val_loader, epochs=EPOCH)
