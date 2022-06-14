import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from config import BS, CODE_MARK_PATH, DATA_DIR, NW, RANK_COUNT, RANKS
from dataset import MarkdownDataset, MarkdownDatasetTest
from helper import get_features_val, kendall_tau
from model import MarkdownModel

device = 'cuda'
torch.cuda.empty_cache()
np.random.seed(0)
torch.manual_seed(0)

model = MarkdownModel().to(device)
model.load_state_dict(torch.load(CODE_MARK_PATH))

df_orders = pd.read_csv(
    DATA_DIR / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

train_df = pd.read_csv('data_dump/train_df.csv')
val_df = pd.read_csv('data_dump/val_df.csv')

train_df_mark = train_df[train_df["cell_type"]
                         == "markdown"].reset_index(drop=True)

with open('data_dump/dict_cellid_source.pkl', 'rb') as f:
    dict_cellid_source = pickle.load(f)
f.close()

with open('data_dump/features_train.pkl', 'rb') as f:
    train_fts = pickle.load(f)
f.close()

unique_ids = pd.unique(val_df['id'])
ids = unique_ids[:1000]
val_df = val_df[val_df['id'].isin(ids)]
val_fts = get_features_val(val_df)

train_ds = MarkdownDataset(train_df_mark, dict_cellid_source,
                           md_max_len=64, total_max_len=512, fts=train_fts)
val_ds = MarkdownDatasetTest(
    dict_cellid_source, md_max_len=64, total_max_len=512, fts=val_fts)

train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=NW,
                          pin_memory=False, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BS * 5, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

EPOCHS = 5
accumulation_steps = 4

num_train_optimization_steps = int(
    EPOCHS * len(train_loader) / accumulation_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                  correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                            num_training_steps=num_train_optimization_steps)  # PyTorch scheduler


def cal_kendall_tau(df, pred):
    index = 0
    df = df.sort_values('rank').reset_index(drop=True)
    df.loc[df['cell_type'] == 'code',
           'pred'] = df[df.cell_type == 'code']['rank']

    for idx, sub_df in tqdm(df.groupby('id')):

        mark_sub_df_all = sub_df[sub_df.cell_type == 'markdown']
        code_sub_df_all = sub_df[sub_df.cell_type == 'code']

        for i in range(0, mark_sub_df_all.shape[0]):
            for j in range(0, code_sub_df_all.shape[0], RANK_COUNT):
                code_sub_df = code_sub_df_all[j: j + RANK_COUNT]
                rank_index = RANKS[pred[index]]
                if rank_index >= 0 and rank_index < 21:
                    if rank_index == 0:
                        df.loc[df['cell_id'] == mark_sub_df_all.iloc[i]
                               ['cell_id'], 'pred'] = 0
                    else:
                        start_rank = 0
                        if rank_index < len(code_sub_df):
                            start_rank = code_sub_df.iloc[rank_index]['rank']
                        df.loc[df['cell_id'] == mark_sub_df_all.iloc[i]
                               ['cell_id'], 'pred'] = start_rank + 1
                index += 1

    df[['id', 'cell_id', 'cell_type', 'rank', 'pred']].to_csv('predict.csv')
    y_dummy = df.sort_values("pred").groupby('id')['cell_id'].apply(list)
    print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))


def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, (ids, mask, fts, target) in enumerate(tbar):
            with torch.cuda.amp.autocast():
                pred = model(ids.to(device), mask.to(device), fts.to(device))
            pred = torch.argmax(pred, dim=1)
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


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

    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        total_loss = 0
        total_step = 0

        for idx, (ids, mask, fts, target) in enumerate(tbar):
            with torch.cuda.amp.autocast():
                pred = model(ids.to(device), mask.to(device), fts.to(device))
                loss = criterion(pred, torch.squeeze(target).to(device))
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

            # if idx % 5000 == 0 or idx == len(tbar) - 1:
            #     label, y_pred = validate(model, val_loader)
            #     cal_kendall_tau(val_df, y_pred)
            #     torch.save(model.state_dict(), CODE_MARK_PATH)
            #     model.train()


model = MarkdownModel()
model = model.cuda()
train(model, train_loader, val_loader, epochs=EPOCHS)
